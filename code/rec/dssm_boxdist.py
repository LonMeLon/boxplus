import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed#, set_color

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

from recbole.model.loss import BPRLoss
import torch.nn.functional as F
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers

class Box(torch.nn.Module):
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

class DSSM(ContextRecommender):
    """ DSSM respectively expresses user and item as low dimensional vectors with mlp layers,
    and uses cosine distance to calculate the distance between the two semantic vectors.
    """

    def __init__(self, config, dataset):
        super(DSSM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        self.user_feature_num = self.user_token_field_num + self.user_float_field_num + self.user_token_seq_field_num
        self.item_feature_num = self.item_token_field_num + self.item_float_field_num + self.item_token_seq_field_num
        user_size_list = [self.embedding_size * self.user_feature_num] + self.mlp_hidden_size
        item_size_list = [self.embedding_size * self.item_feature_num] + self.mlp_hidden_size

        # define layers and loss
        self.user_mlp_layers = MLPLayers(user_size_list, self.dropout_prob, activation='tanh', bn=True)
        self.item_mlp_layers = MLPLayers(item_size_list, self.dropout_prob, activation='tanh', bn=True)

        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # box parameter
        self.scale = 1.0
        ### query mlp layers
        self.layer_query_min = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1])
        )
        self.layer_query_delta = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1])
        )
        ### doc mlp layers
        self.layer_doc_min = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1]),
            torch.nn.LeakyReLU(), 
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1])
        )
        self.layer_doc_delta = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.mlp_hidden_size[-1], self.mlp_hidden_size[-1])
        )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        # user_sparse_embedding shape: [batch_size, user_token_seq_field_num + user_token_field_num , embed_dim] or None
        # user_dense_embedding shape: [batch_size, user_float_field_num] or [batch_size, user_float_field_num, embed_dim] or None
        # item_sparse_embedding shape: [batch_size, item_token_seq_field_num + item_token_field_num , embed_dim] or None
        # item_dense_embedding shape: [batch_size, item_float_field_num] or [batch_size, item_float_field_num, embed_dim] or None
        embed_result = self.double_tower_embed_input_fields(interaction)
        user_sparse_embedding, user_dense_embedding = embed_result[:2]
        item_sparse_embedding, item_dense_embedding = embed_result[2:]

        user = []
        if user_sparse_embedding is not None:
            user.append(user_sparse_embedding)
        if user_dense_embedding is not None and len(user_dense_embedding.shape) == 3:
            user.append(user_dense_embedding)

        embed_user = torch.cat(user, dim=1)

        item = []
        if item_sparse_embedding is not None:
            item.append(item_sparse_embedding)
        if item_dense_embedding is not None and len(item_dense_embedding.shape) == 3:
            item.append(item_dense_embedding)

        embed_item = torch.cat(item, dim=1)

        batch_size = embed_item.shape[0]
        user_dnn_out = self.user_mlp_layers(embed_user.view(batch_size, -1))
        item_dnn_out = self.item_mlp_layers(embed_item.view(batch_size, -1))


        scores = self.get_score_boxdist(user_dnn_out, item_dnn_out)
        
        predict_scores = 10 * (1 + 3 * torch.tanh(-scores))  # 10 * (-2, 1]
        predict_scores = self.sigmoid(predict_scores)
        return predict_scores
        #score = torch.cosine_similarity(user_dnn_out, item_dnn_out, dim=1)

        #sig_score = self.sigmoid(score)
        #return sig_score.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)



    def get_score_boxdist(self, user_dnn_out, item_dnn_out):
        # box bound
        user_box_low, user_box_up = self.get_bound_from_vec(user_dnn_out, "query")
        item_box_low, item_box_up = self.get_bound_from_vec(item_dnn_out, "item")
        user_box = Box(user_box_low, user_box_up)
        item_box = Box(item_box_low, item_box_up)
        # intersection
        intersect_min = torch.max(user_box.min_embed, item_box.min_embed)
        intersect_max = torch.min(user_box.max_embed, item_box.max_embed)
        intersect_box = Box(intersect_min, intersect_max)
        # boxdist
        boxLens = intersect_box.max_embed - intersect_box.min_embed
        score = torch.norm(F.relu(-boxLens), p=2, dim=1) 
        
        return score

    def get_bound_from_vec(self, dnn_out, mark):
        if mark == "query":
            box_low = self.layer_query_min(dnn_out)
            #box_low = torch.exp(self.layer_query_min(dnn_out))
            box_up = box_low + 0.1 * torch.exp(self.layer_query_delta(dnn_out))
        if mark == "item":
            box_low = self.layer_doc_min(dnn_out)
            #box_low = torch.exp(self.layer_doc_min(dnn_out))
            box_up = box_low + 0.1 * torch.exp(self.layer_doc_delta(dnn_out))
        return box_low, box_up


def run_single_model(args):
    # configurations initialization
    '''
    config_dict = {
    'train_stage': 'pretrain',
    'save_step': 10,
    'data_path' : '../data/'
    }
    '''

    config = Config(
        model=args.model,    # RecBole requires a str model name, GRU4Rec just as placeholder
        dataset=args.dataset, 
        config_file_list=[args.yaml]
    )
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = DSSM(config, train_data).to(config['device'])
    print('device', next(model.parameters()).device) 
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    #checkpoint_file = 'saved/SASRec-Oct-10-2021_13-15-00.pth'
    #trainer.resume_checkpoint(checkpoint_file)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DSSM')
    parser.add_argument('--dataset', type=str, default='Amazon_Books')
    parser.add_argument('--yaml', type=str, default='../amazon-books.yaml')
    args, _ = parser.parse_known_args()

    run_single_model(args)
    
    
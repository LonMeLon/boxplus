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
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MLPLayers

class Box(torch.nn.Module):
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

class GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.
    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # box parameter
        self.scale = 1.0
        ### query mlp layers
        self.layer_query_min = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size)
        )
        self.layer_query_delta = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size)
        )
        ### doc mlp layers
        self.layer_doc_min = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size)
        )
        self.layer_doc_delta = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size)
        )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            #pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            #neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            pos_score = self.get_score_boxdist(seq_output, pos_items_emb)
            pos_score = self.small_change(pos_score)
            neg_score = self.get_score_boxdist(seq_output, neg_items_emb)
            neg_score = self.small_change(neg_score)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)

        scores = self.get_score_boxdist(seq_output, test_item_emb)
        scores = self.small_change(scores)
        #scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def small_change(self, scores):
        #return -scores + 20 # (-00, 20)
        return 10 * (1 + 3 * torch.tanh(-scores))  # 10 * (-2, 1]

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
    model = GRU4Rec(config, train_data).to(config['device'])
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
    parser.add_argument('--model', type=str, default='GRU4Rec')
    parser.add_argument('--dataset', type=str, default='Amazon_Books')
    parser.add_argument('--yaml', type=str, default='../amazon-books.yaml')
    args, _ = parser.parse_known_args()

    run_single_model(args)
    
    

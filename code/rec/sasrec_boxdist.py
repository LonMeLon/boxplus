import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed#, set_color

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

class Box(torch.nn.Module):
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

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
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.layer_query_delta = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        ### doc mlp layers
        self.layer_doc_min = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(), 
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.layer_doc_delta = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

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
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
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
    model = SASRec(config, train_data).to(config['device'])
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
    parser.add_argument('--model', type=str, default='SASRec')
    parser.add_argument('--dataset', type=str, default='Amazon_Books')
    parser.add_argument('--yaml', type=str, default='../amazon-books.yaml')
    args, _ = parser.parse_known_args()

    run_single_model(args)
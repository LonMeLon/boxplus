# coding:utf8

from numpy.core.einsumfunc import _optimal_path
import torch
import torch.nn as nn
import random
import sys
import numpy as np
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)
import time
import gzip
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from datetime import datetime
from torch.distributions import uniform
#from scipy.stats import pearsonr, spearmanr

sys.path.append(".")
sys.path.append("/nfs/volume-93-2/meilang/myprojects/box_optimize/wordnet/")

from basic_box import Box

euler_gamma = 0.57721566490153286060

class GradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # [batch, dim]
        return torch.sum(torch.log(input), dim=-1) + torch.log(torch.tensor(1.0))

    @staticmethod
    def backward(ctx, grad_chain):
        input, = ctx.saved_tensors
        # = 1 / X_i [batch, dim]
        log_FX = torch.sum(torch.log(input), dim=-1) + torch.log(torch.tensor(1.0))
        FX = torch.exp(log_FX) #[batch, ]
        grad_input = 1. / input
        max_grad_input = torch.max(grad_input, 1).values.reshape(-1, 1).repeat(1, input.shape[1])
        norm_grad_input = (grad_input / max_grad_input) * (1e-2 / FX).reshape(-1, 1).repeat(1, input.shape[1])

        #print(norm_grad_input.shape, grad_chain.shape)
        return norm_grad_input * grad_chain.reshape(-1, 1).repeat(1, input.shape[1])
        '''
        # grad: dlogp / dx; (1 / p) * grad: dp / dx
        margin =  torch.finfo(torch.FloatTensor([1.0]).dtype).max  # type: ignore
        log_margin = float(int(np.log(margin)))
        ### 1 / p
        ValueIntersectDist = torch.sum(torch.log(input), dim=-1) + torch.log(torch.tensor(1.0))
        one = torch.exp((-ValueIntersectDist).clamp_max(log_margin))
        #print("one", one)
        ### grad: dp / dx
        two = (1.0 / one).reshape(-1, 1).repeat(1, input.shape[1]) / input
        #two_norm = torch.min(input, -1).values.reshape(-1, 1).repeat(1, input.shape[1]) / input  #two  / torch.max(two, -1).values.reshape(-1, 1).repeat(1, two.shape[1])
        two_norm = two.clamp_min(1e-3)
        ###
        #print(grad_output.shape)
        #print(one.shape, two.shape)
        grad_input = grad_output.reshape(-1, 1).repeat(1, input.shape[1]) * one.reshape(-1, 1).repeat(1, input.shape[1]) * two_norm
        #grad_input = grad_output * (-torch.exp(-input)/input)
        '''
        #return grad_input

class GumbelBoxDist(nn.Module):
    def __init__(self, args):
        super(GumbelBoxDist, self).__init__()
        self.args = args
        min_embedding = self.init_word_embedding(args["movie_size"], args["output_dim"], args["init_min"])
        delta_embedding = self.init_word_embedding(args["movie_size"], args["output_dim"], args["init_delta"])
        #self.temperature = args.softplus_temp
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)
        #self.gumbel_beta = args.gumbel_beta
        self.scale = args["scale"]

        self.gradnorm = GradNorm.apply

    def forward(self, ids):
        """Returns box embeddings for ids"""
        min_rep = self.min_embedding[ids]
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep + torch.exp(delta_rep)
        # print('min', min_rep.mean())
        # print('delta', torch.exp(delta_rep.mean()))
        # wandb.log(self.min_embedding.mean())
        # wandb.log(self.delta_embedding.mean())
        boxes1 = Box(min_rep[:, 0, :], max_rep[:, 0, :])
        boxes2 = Box(min_rep[:, 1, :], max_rep[:, 1, :])
        Index_overlap, Index_disjoint, pos_predictions = self.get_cond_probs(boxes1, boxes2)
        neg_prediction = torch.ones(pos_predictions.size()).to('cuda:0') - pos_predictions
        prediction = torch.stack([neg_prediction, pos_predictions], dim=1)
        return Index_overlap, Index_disjoint, prediction

    def all_boxes_volumes(self, ):
        min_rep = self.min_embedding
        delta_rep = self.delta_embedding
        max_rep = min_rep + torch.exp(delta_rep)
        all_boxes = Box(min_rep, max_rep)
        _, _, predictions = self.get_cond_probs(all_boxes, all_boxes)
        return 0.01 * predictions.mean()


    def volumes(self, boxes, scale = 1.):
        eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

        if isinstance(self.scale, float):
            s = torch.tensor(self.scale)
        else:
            s = self.scale
        # overlap mark
        ### origin box
        boxLens = boxes.max_embed - boxes.min_embed
        ### gumbel box
        #boxLens = boxes.max_embed - boxes.min_embed - 2 * euler_gamma * self.gumbel_beta

        Tag = (torch.min(boxLens, dim=1).values > 0.0)
        Index_overlap = torch.where(Tag == True)
        Index_disjoint = torch.where(Tag == False)
        #if Index_disjoint[0].shape[0] > 0:
        #    print("111")
        # meansure (prob or distance)
        pnorm = 2
        measures = torch.zeros(boxLens.shape[0]).to('cuda:0')

        if Index_overlap[0].shape[0] > 0:
            #measures[Index_overlap] = torch.sum(torch.log(F.relu(boxLens[Index_overlap]).clamp_min(eps)), dim=-1) + torch.log(s)
            measures[Index_overlap] = self.gradnorm(F.relu(boxLens[Index_overlap]).clamp_min(eps))
        if Index_disjoint[0].shape[0] > 0:
            measures[Index_disjoint] = torch.norm(F.relu(-boxLens[Index_disjoint]), p=pnorm, dim=1) 

        return Index_overlap, Index_disjoint, measures

    def intersection(self, boxes1, boxes2):
        ### origin box
        z = torch.max(boxes1.min_embed, boxes2.min_embed)
        Z = torch.min(boxes1.max_embed, boxes2.max_embed)
        '''
        ### gumbel box
        z = self.gumbel_beta * torch.logsumexp(torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)), 0)
        z = torch.max(z, torch.max(boxes1.min_embed, boxes2.min_embed)) # This line is for numerical stability (you could skip it if you are not facing any issues)

        Z = - self.gumbel_beta * torch.logsumexp(torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)), 0)
        Z = torch.min(Z, torch.min(boxes1.max_embed, boxes2.max_embed)) # This line is for numerical stability (you could skip it if you are not facing any issues)
        '''

        intersection_box = Box(z, Z)
        return intersection_box

    def get_cond_probs(self, boxes1, boxes2):
        # log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        # log_box2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        '''
        log_intersection = torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), np.log(1e-10), np.log(1e4))
        log_box2 = torch.clamp(self.volumes(boxes2), np.log(1e-10), np.log(1e4))
        return torch.exp(log_intersection-log_box2)
        '''

        Index_overlap, Index_disjoint, log_inter_measure = self.volumes(self.intersection(boxes1, boxes2))
        _, _, log_box2 = self.volumes(boxes2)
        measures = torch.zeros(log_inter_measure.shape[0]).to('cuda:0')
        if Index_overlap[0].shape[0] > 0:
            measures[Index_overlap] = torch.exp(log_inter_measure[Index_overlap] - log_box2[Index_overlap])
        if Index_disjoint[0].shape[0] > 0:
            measures[Index_disjoint] = log_inter_measure[Index_disjoint]

        return Index_overlap, Index_disjoint, measures

    def init_word_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {
        "method" : "train",

        "batch_size" : 100 * 100 * 2,
        'movie_size': 100,  
        'output_dim': 50, 
        "init_min" : [1e-4, 0.5], #[1e-4, 0.2],
        "init_delta": [-0.05, 0.05], #[-0.7, -0.6],#[-0.1, -0.0],

        "scale" : 1.0,
        "epoch" : 10000,
        "learning_rate" : 0.1,
        "device" : device
    }

    print(args)
    
    if args['method'] == 'train':
        train_AB_condprob = np.load("train_AB_condprob.npy")
        test_AB_condprob = np.load("test_AB_condprob.npy")

        # network structure
        model_lstm_box = GumbelBoxDist(args).to(args["device"])
        criterion = torch.nn.CrossEntropyLoss().to(device)

        t1 = datetime.now()
        # input train data
        ### sequence data
        input_AB = torch.LongTensor(train_AB_condprob.T[0:2].T)
        input_conprob = torch.LongTensor(np.zeros_like(train_AB_condprob.T[2])) # more disjoint
        
        #input_conprob = torch.LongTensor(np.ones_like(train_AB_condprob.T[2])) # more overlap
        #print(torch.where(input_conprob == 1))
        t2 = datetime.now()
        print((t2 - t1).seconds)

        # train      
        ### random
        all_indexs = np.array([i for i in range(input_conprob.shape[0])])
        all_indexs_random = torch.LongTensor(np.random.permutation(all_indexs))
        all_boxes_indexes = torch.LongTensor([i for i in range(args["movie_size"])]).to(args["device"])
        t2 = datetime.now()
        print((t2 - t1).seconds)
        ### setting
        KL_loss_function = nn.KLDivLoss(size_average=True, reduce=True)
        MSE_loss_function = nn.MSELoss()
        CE_loss_function = nn.CrossEntropyLoss()
        NLL_loss_function = nn.NLLLoss(reduction="none")
        optimizer = torch.optim.SGD(model_lstm_box.parameters(), lr=args["learning_rate"])
        ### epoch
        for epoch in tqdm(range(args["epoch"])):
            input_AB_step = input_AB.to(args["device"])
            input_condprob_step = input_conprob.to(args["device"])
            #print(input_AB.shape, input_conprob.shape[0])
            #print(input_AB_step.shape, input_condprob_step.shape[0])

            ### see all training pair state
            Index_overlap, Index_disjoint, output = model_lstm_box(input_AB_step)
            
            if (epoch + 1) % 100 == 0:
                print('epoch: ', epoch)
                print("info: ", "total:", input_AB_step[0].shape[0], "overlap:", Index_overlap[0].shape[0], "disjoint:", Index_disjoint[0].shape[0])
                print("ratio overlap: ", Index_overlap[0].shape[0] / input_AB_step.shape[0])
                print("ratio disjoint: ", Index_disjoint[0].shape[0] / input_AB_step.shape[0])

            ### training
            optimizer.zero_grad()

            ### model
            Index_overlap, Index_disjoint, output = model_lstm_box(input_AB_step)
            if Index_overlap[0].shape[0] > 0:
                loss_overlap = criterion(output[Index_overlap], input_condprob_step[Index_overlap])
            else:
                loss_overlap = torch.zeros(1).to('cuda:0')
            if Index_disjoint[0].shape[0] > 0:
                loss_disjoint = (input_condprob_step[Index_disjoint] * output[Index_disjoint][:, 1]).mean()
                #print(Index_disjoint, loss_disjoint)
            else:
                loss_disjoint = torch.zeros(1).to('cuda:0')
            loss_regular = model_lstm_box.all_boxes_volumes()

            loss = loss_overlap + loss_disjoint + loss_regular

            # backward and optimize
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print('loss: ', loss.item(), loss_overlap.item(), loss_disjoint.item(), loss_regular.item())
            

        

            torch.save({'args': args,
                            'epoch': epoch,
                            'state_dict': model_lstm_box.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            'checkpoint-box/checkpoint' + str(epoch) +'.pth')

    elif args['method'] == 'test': # predict probabilities on test file (probability format)
        ckpt = torch.load("checkpoint-soft/checkpoint90.pth")
        args = ckpt['args']

        test_AB_condprob = np.load("test_AB_condprob.npy")

        # network structure
        model_lstm_box = BoxDist(args).to(args["device"])
        model_lstm_box.load_state_dict(ckpt['state_dict'])
        model_lstm_box.eval()
        model_lstm_box.to(args["device"])

        total_loss = 0.0
        pearsonr_score = [[], []]
        spearmanr_score = [[], []]

        KL_loss_function = nn.KLDivLoss(size_average=True, reduce=True)
        ### sequence data
        input_AB = torch.LongTensor(test_AB_condprob.T[0:2].T)
        input_conprob = torch.FloatTensor(test_AB_condprob.T[2])
        ### batch
        for step in range(int(test_AB_condprob.shape[0] / args["batch_size"]) + 1):
            ### indices
        
            ### step input
            ##### sequence data
            input_AB_step = input_AB[step * args["batch_size"] : (step + 1) * args["batch_size"]].to(args["device"])
            input_condprob_step = input_conprob[step * args["batch_size"] : (step + 1) * args["batch_size"]].to(args["device"])

            ### model
            ##### get box
            boxes1_minbound, boxes1_maxbound = model_lstm_box.get_box(input_AB_step[:, 0])
            boxes2_minbound, boxes2_maxbound = model_lstm_box.get_box(input_AB_step[:, 1])
            ##### get prob
            log_x_preds, log_y_preds, log_cpr_preds = model_lstm_box.get_prob_output(boxes1_minbound, boxes1_maxbound, boxes2_minbound, boxes2_maxbound)
            
            ### loss: x, y, cpr: KL
            #log_x_preds_neg = torch.log(1 - torch.exp(log_x_preds))
            #log_y_preds_neg = torch.log(1 - torch.exp(log_y_preds))
            log_cpr_preds_neg = torch.log(1 - torch.exp(log_cpr_preds))

            #log_cat_x_preds = torch.cat([log_x_preds.reshape(-1, 1), log_x_preds_neg.reshape(-1, 1)], dim=1)
            #log_cat_y_preds = torch.cat([log_y_preds.reshape(-1, 1), log_y_preds_neg.reshape(-1, 1)], dim=1)
            log_cat_cpr_preds = torch.cat([log_cpr_preds.reshape(-1, 1), log_cpr_preds_neg.reshape(-1, 1)], dim=1)

            #cat_input_train_xlabels_step = torch.cat([input_train_xlabels_step.reshape(-1, 1), (1 - input_train_xlabels_step).reshape(-1, 1)], dim=1)
            #cat_input_train_ylabels_step = torch.cat([input_train_ylabels_step.reshape(-1, 1), (1 - input_train_ylabels_step).reshape(-1, 1)], dim=1)
            cat_input_train_cpr_labels_step = torch.cat([input_condprob_step.reshape(-1, 1), (1 - input_condprob_step).reshape(-1, 1)], dim=1)


            #x_loss = KL_loss_function(log_cat_x_preds, cat_input_train_xlabels_step) 
            #y_loss = KL_loss_function(log_cat_y_preds, cat_input_train_ylabels_step) 
            cpr_loss = KL_loss_function(log_cat_cpr_preds, cat_input_train_cpr_labels_step) 
            
            
            #x_loss = MSE_loss_function(log_x_preds.exp(), input_train_xlabels_step) 
            #y_loss = MSE_loss_function(log_y_preds.exp(), input_train_ylabels_step) 
            #cpr_loss = MSE_loss_function(log_cpr_preds.exp(), input_train_cpr_labels_step) 
            

            total_loss += cpr_loss.item() * input_AB_step.shape[0]
            pearsonr_score[0] += list(input_condprob_step.detach().cpu().numpy())
            pearsonr_score[1] += list(log_cpr_preds.exp().detach().cpu().numpy())
            
            spearmanr_score[0] += list(input_condprob_step.detach().cpu().numpy())
            spearmanr_score[1] += list(log_cpr_preds.exp().detach().cpu().numpy())
        
        print(np.array(pearsonr_score[0]).shape)
        print(pearsonr(pearsonr_score[0], pearsonr_score[1]))
        print(spearmanr(spearmanr_score[0], spearmanr_score[1]))
        
        print(total_loss / test_AB_condprob.shape[0])

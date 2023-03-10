import torch
#import wandb
import numpy as np
import torch.nn as nn
from basic_box import Box
#from utils import ExpEi
import torch.nn.functional as F
from torch.distributions import uniform

euler_gamma = 0.57721566490153286060

class GradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
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

class GumbelBoxDist(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, min_init_value, delta_init_value, args):
        super(GumbelBoxDist, self).__init__()
        self.args = args
        min_embedding = self.init_word_embedding(vocab_size, embed_dim, min_init_value)
        delta_embedding = self.init_word_embedding(vocab_size, embed_dim, delta_init_value)
        self.temperature = args.softplus_temp
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)
        self.gumbel_beta = args.gumbel_beta
        self.scale = args.scale

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
        boxLens = boxes.max_embed - boxes.min_embed - 2 * euler_gamma * self.gumbel_beta

        Tag = (torch.min(boxLens, dim=1).values > 0.0)
        Index_overlap = torch.where(Tag == True)
        Index_disjoint = torch.where(Tag == False)
        #if Index_disjoint[0].shape[0] > 0:
        #    print("111")
        # meansure (prob or distance)
        pnorm = 2
        measures = torch.zeros(boxLens.shape[0]).to('cuda:0')

        if Index_overlap[0].shape[0] > 0:
            measures[Index_overlap] = torch.sum(torch.log(F.relu(boxLens[Index_overlap]).clamp_min(eps)), dim=-1) + torch.log(s)
            #measures[Index_overlap] = self.gradnorm(F.relu(boxLens[Index_overlap]).clamp_min(eps))
        if Index_disjoint[0].shape[0] > 0:
            measures[Index_disjoint] = torch.norm(F.relu(-boxLens[Index_disjoint]), p=pnorm, dim=1) 

        return Index_overlap, Index_disjoint, measures

    def intersection(self, boxes1, boxes2):
        z = self.gumbel_beta * torch.logsumexp(torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)), 0)
        z = torch.max(z, torch.max(boxes1.min_embed, boxes2.min_embed)) # This line is for numerical stability (you could skip it if you are not facing any issues)

        Z = - self.gumbel_beta * torch.logsumexp(torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)), 0)
        Z = torch.min(Z, torch.min(boxes1.max_embed, boxes2.max_embed)) # This line is for numerical stability (you could skip it if you are not facing any issues)

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


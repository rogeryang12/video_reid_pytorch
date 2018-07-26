import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TripletLoss(nn.Module):
    def __init__(self, margin, mode='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode

    def forward(self, embs, pids):
        b, emb_dim = embs.size()
        dists = torch.norm(embs.view(b, 1, -1) - embs.view(1, b, -1), dim=-1)

        pos_mask = torch.eq(pids.view(b, 1), pids.view(1, b))
        pos_mask = pos_mask ^ torch.eye(b).type_as(pos_mask)
        neg_mask = torch.ne(pids.view(b, 1), pids.view(1, b))
        if self.mode == 'uniform':
            pos_dist = torch.stack([torch.mean(dists[i][pos_mask[i]]) for i in range(b)])
            neg_dist = torch.stack([torch.mean(dists[i][neg_mask[i]]) for i in range(b)])
        elif self.mode == 'hard':
            pos_dist = torch.stack([torch.max(dists[i][pos_mask[i]]) for i in range(b)])
            neg_dist = torch.stack([torch.min(dists[i][neg_mask[i]]) for i in range(b)])
        elif self.mode == 'adaptive':
            pos_dist = torch.stack([self.weight_sum(dists[i][pos_mask[i]], 1) for i in range(b)])
            neg_dist = torch.stack([self.weight_sum(dists[i][neg_mask[i]], -1) for i in range(b)])

        diff = pos_dist - neg_dist
        if self.margin == 'soft':
            loss = torch.mean(nn.Softplus()(diff))
        else:
            loss = torch.mean(torch.clamp(diff + self.margin, min=0))

        pids = pids.cpu().detach().numpy()
        _, index = torch.topk(-dists, 2, dim=-1)
        top1 = np.mean(pids == pids[index[:, 1].cpu().numpy()])

        return loss, top1

    def weight_sum(self, dist, scalar):
        weight = F.softmax(dist * scalar, dim=-1)
        return (dist * weight).sum()


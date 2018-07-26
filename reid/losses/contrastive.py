import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss, return loss
    """
    def __init__(self, margin=2, batch_first=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch_first = batch_first

    def forward(self, embs, same):
        """
        :param embs: batch_first==True: batch_size*2*emb_dim; batch_first==False: 2*batch_size*emb_dim
        :param same: batch_size, whether or not 2 embedding belong to same person
        :return:
        """
        if self.batch_first:
            embs = embs.transpose(0, 1).contiguous()
        dist = (embs[0] - embs[1]).pow(2).sum(dim=-1)
        m_dist = torch.clamp(self.margin - dist, min=0)
        same = same.type_as(dist)
        loss = same * dist + (1 - same) * m_dist
        loss = loss.mean() / 2
        return loss

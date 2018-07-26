import torch.nn as nn

from .utils import CNNBackbone


class QAN(nn.Module):
    """
    CNN model return embedding or embedding and logits
    """
    def __init__(self, emb_dim=1024, num_classes=None, cnn='resnet50', ckpt=None):
        """
        :param emb_dim:  dimension of embedding
        :param num_classes:  number of classes
        :param cnn:  cnn model option
        :param ckpt:  pre-trained checkpoint of cnn model
        """
        super(QAN, self).__init__()
        self.backbone = CNNBackbone(cnn, ckpt, pool='avg', dropout=True)
        self.embedding = nn.Sequential(nn.Linear(2048, emb_dim), nn.BatchNorm1d(emb_dim))
        self.quality = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
        self.num_calsses = num_classes
        if num_classes is not None:
            self.logits = nn.Sequential(nn.Dropout(),
                                        nn.ReLU(),
                                        nn.Linear(emb_dim, num_classes))

    def forward(self, x):
        if self.training:
            # shape of x: batch_size*image_num*channel*height*weight
            b, t, c, h, w = x.size()
            x = self.backbone(x.view(-1, c, h, w))
            qual = self.quality(x).view(b, t, 1)
            emb = self.embedding(x)
            v_emb = (qual * emb.view(b, t, -1)).sum(dim=1) / (qual.sum(dim=1) + 1e-12)
            if self.num_calsses is not None:
                logit = self.logits(emb)
                return emb, v_emb, logit
            return emb, v_emb
        else:
            # shape of x: batch_size*channel*height*weight
            x = self.backbone(x)
            emb = self.embedding(x)
            qual = self.quality(x)
            return emb, qual

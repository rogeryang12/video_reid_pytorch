import torch.nn as nn

from .utils import CNNBackbone


class CNN(nn.Module):
    """
    CNN model return embedding (and logits).
    """
    def __init__(self, emb_dim=1024, num_classes=None, cnn='resnet50', ckpt=None):
        """
        :param emb_dim:  dimension of embedding
        :param num_classes:  number of classes
        :param cnn:  cnn model option
        :param ckpt:  pre-trained checkpoint of cnn model
        """
        super(CNN, self).__init__()
        self.backbone = CNNBackbone(cnn, ckpt, pool='avg', dropout=True)
        self.embedding = nn.Sequential(nn.Linear(2048, emb_dim), nn.BatchNorm1d(emb_dim))
        self.num_calsses = num_classes
        if num_classes is not None:
            self.logits = nn.Sequential(nn.Dropout(),
                                        nn.ReLU(),
                                        nn.Linear(emb_dim, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        emb = self.embedding(x)
        if self.num_calsses is not None:
            logit = self.logits(emb)
            return emb, logit
        return emb

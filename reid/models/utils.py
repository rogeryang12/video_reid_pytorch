import torch
import torch.nn as nn
import torchvision


class CNNBackbone(nn.Module):

    def __init__(self, cnn='resnet50', ckpt=None, pool='avg', dropout=True):
        """
        cnn that return feature maps or feature vectors (global pooling)
        :param cnn: cnn model option
        :param ckpt: pre-trained checkpoint of cnn model
        :param pool: pool the feature maps. choices: avg, max, None
        :param dropout: whether or not add dropout
        """
        super(CNNBackbone, self).__init__()
        self.transform_input = False
        self.pool = pool
        if cnn == 'inception3':
            models = torchvision.models.inception_v3(aux_logits=False)
            if ckpt is not None:
                models.load_state_dict(torch.load(ckpt), False)
                self.transform_input = True
            models = list(models.children())[:-1]

        elif cnn == 'resnet50':
            models = torchvision.models.resnet50()
            if ckpt is not None:
                models.load_state_dict(torch.load(ckpt))
            models = list(models.children())[:-2]

        if pool is not None:
            if pool == 'avg':
                models.append(nn.AdaptiveAvgPool2d(1))
            elif pool == 'max':
                models.append(nn.AdaptiveMaxPool2d(1))
        if dropout:
            models.append(nn.Dropout())

        self.model = nn.Sequential(*models)

    def forward(self, x):
        if self.transform_input:
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.model(x)
        if self.pool is not None:
            x = x.view(x.size(0), -1)
        return x



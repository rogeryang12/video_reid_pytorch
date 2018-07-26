import torch.nn as nn

from . import CNN


class RCN(nn.Module):

    def __init__(self, emb_dim=1024, num_classes=None, cnn='resnet50', ckpt=None, norm=True):
        super(RCN, self).__init__()
        self.cnn = CNN(emb_dim, num_classes, cnn, ckpt)
        self.rnn = nn.RNN(emb_dim, emb_dim, batch_first=True)
        self.norm = norm
        self.num_classes = num_classes

    def forward(self, x):
        if self.training:
            # shape of x: batch_size*image_num*channel*height*weight
            b, t, c, h, w = x.size()
            out = self.cnn(x.view(-1, c, h, w))
            emb = out if self.num_classes is None else out[0]
            x, _ = self.rnn(emb.view(b, t, -1))
            x = x.mean(dim=1)
            v_emb = x / (x.norm(dim=-1, keepdim=True) + 1e-12) if self.norm else x
            if self.num_classes is not None:
                return emb, v_emb, out[1]
            return emb, v_emb
        else:
            x = self.cnn(x)
            return x

    def cal_video_emb(self, x):
        # shape of x: image_num*emb_dim
        x = x.view(1, *x.size())
        x, _ = self.rnn(x)
        x = x.mean(dim=1).view(-1)
        v_emb = x / (x.norm() + 1e-12) if self.norm else x
        return v_emb


class TCN(nn.Module):

    def __init__(self, channel=32, kernel=4, emb_dim=1024, num_classes=None, cnn='resnet50', ckpt=None, norm=True):
        super(TCN, self).__init__()
        self.cnn = CNN(emb_dim, num_classes, cnn, ckpt)
        self.tcn = nn.Sequential(nn.Conv1d(1, channel, kernel),
                                 nn.ReLU(),
                                 nn.Conv1d(channel, 1, 1),
                                 nn.ReLU())
        self.rnn = nn.RNN(emb_dim, emb_dim, batch_first=True)
        self.norm = norm
        self.emb_dim = emb_dim
        self.num_classes = num_classes

    def forward(self, x):
        if self.training:
            # shape of x: batch_size*image_num*channel*height*weight
            b, t, c, h, w = x.size()
            out = self.cnn(x.view(-1, c, h, w))
            emb = out if self.num_classes is None else out[0]
            x = emb.view(b, t, -1).transpose(1, 2).contiguous()
            x = self.tcn(x.view(-1, 1, t))
            x = x.view(b, self.emb_dim, -1).transpose(1, 2).contiguous()
            x, _ = self.rnn(x)
            x = x.mean(dim=1)
            v_emb = x / (x.norm(dim=-1, keepdim=True) + 1e-12) if self.norm else x
            if self.num_classes is not None:
                return emb, v_emb, out[1]
            return emb, v_emb
        else:
            x = self.cnn(x)
            return x

    def cal_video_emb(self, x):
        # shape of x: image_num*emb_dim
        t, emb_dim = x.size()
        x = x.transpose(0, 1).contiguous().view(emb_dim, 1, t)
        x = self.tcn(x)
        x = x.view(1, emb_dim, -1).transpose(1, 2).contiguous()
        x, _ = self.rnn(x)
        x = x.mean(dim=1).view(-1)
        v_emb = x / (x.norm() + 1e-12) if self.norm else x
        return v_emb

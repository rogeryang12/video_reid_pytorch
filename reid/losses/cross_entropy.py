import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    return loss and precision
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.crossentropy(inputs, targets)
        _, predict = torch.max(inputs, dim=1)
        prec = (predict == targets).double().mean().item()
        return loss, prec

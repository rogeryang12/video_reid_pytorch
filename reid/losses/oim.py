
""" OIM loss https://github.com/Cysu/open-reid"""

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F


class OIM(autograd.Function):

    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1 - self.momentum) * x
            self.lut[y] /= (self.lut[y].norm() + 1e-12)
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):

    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs = inputs * self.scalar
        loss = F.cross_entropy(inputs, targets, self.weight, self.size_average)
        _, predict = torch.max(inputs, dim=1)
        prec = (predict == targets).double().mean().item()
        return loss, prec

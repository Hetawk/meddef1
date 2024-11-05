#!/usr/bin/env python3
# utils.py

import torch
from torch import nn


def logsumexp_2d(tensor):
    """LogSumExp for 2D tensor."""
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Flatten(nn.Module):
    """Flatten the input tensor."""
    def forward(self, x):
        return x.view(x.size(0), -1)

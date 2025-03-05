# soft_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention

class SoftAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, **kwargs):
        super(SoftAttention, self).__init__(query_dim, key_dim, value_dim, num_heads)
        self.scale = (key_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)

        # Weighted sum of values
        output = torch.matmul(attn_weights, value)
        output = self.combine_heads(output, batch_size)
        output = self.dense(output)
        return output, attn_weights


# local_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention

class LocalAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, window_size=3, **kwargs):
        super(LocalAttention, self).__init__(query_dim, key_dim, value_dim, num_heads)
        self.window_size = window_size
        self.scale = (key_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        half_window = self.window_size // 2

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        # Pad key and value for local window
        key_padded = F.pad(key, (0, 0, half_window, half_window), "constant", 0)
        value_padded = F.pad(value, (0, 0, half_window, half_window), "constant", 0)

        # Extract local windows
        key_local = key_padded.unfold(2, self.window_size, 1)
        value_local = value_padded.unfold(2, self.window_size, 1)

        # Compute attention scores
        scores = torch.matmul(query, key_local.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unfold(2, self.window_size, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, value_local)
        output = self.combine_heads(output, batch_size)
        output = self.dense(output)
        return output, attn_weights

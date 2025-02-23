# hard_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention

class HardAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, top_k=1, **kwargs):
        super(HardAttention, self).__init__(query_dim, key_dim, value_dim, num_heads)
        self.top_k = top_k
        self.scale = (key_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Select top-k attention scores
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)

        # Create a hard mask
        hard_mask = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)

        # Apply mask to values
        attn_weights = self.softmax(hard_mask)
        output = torch.matmul(attn_weights, value)
        output = self.combine_heads(output, batch_size)
        output = self.dense(output)
        return output, attn_weights

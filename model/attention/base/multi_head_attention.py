
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention


class MultiHeadAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8, **kwargs):
        super(MultiHeadAttention, self).__init__(query_dim, key_dim, value_dim, num_heads, **kwargs)
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.scale = (key_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, value)
        output = self.combine_heads(output, batch_size)
        output = self.dense(output)
        return output, attn_weights
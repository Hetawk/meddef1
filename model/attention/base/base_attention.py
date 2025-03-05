# base_attention.py

import torch
import torch.nn as nn

class BaseAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, **kwargs):
        super(BaseAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.depth = query_dim // num_heads

        self.wq = nn.Linear(query_dim, query_dim)
        self.wk = nn.Linear(key_dim, key_dim)
        self.wv = nn.Linear(value_dim, value_dim)
        self.dense = nn.Linear(value_dim, value_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def combine_heads(self, x, batch_size):
        """Combine the heads back to the original shape."""
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.num_heads * self.depth)

    def forward(self, query, key, value, mask=None):
        raise NotImplementedError("Forward method not implemented in BaseAttention.")

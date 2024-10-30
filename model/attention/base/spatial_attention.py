import torch
import torch.nn as nn
from .base_attention import BaseAttention


class SpatialAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, **kwargs):
        super(SpatialAttention, self).__init__(query_dim, key_dim, value_dim, num_heads, **kwargs)
        self.conv = nn.Conv2d(in_channels=value_dim, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, key, value, mask=None):
        batch_size, channels, height, width = value.size()

        # Apply convolution to get spatial attention map
        attn_map = self.conv(value)
        attn_map = self.sigmoid(attn_map)

        # Apply attention map to the value
        output = value * attn_map

        return output, attn_map

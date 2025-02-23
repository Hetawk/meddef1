import torch
import torch.nn as nn
from .base_attention import BaseAttention

class SpatialAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, **kwargs):
        super(SpatialAttention, self).__init__(query_dim, key_dim, value_dim, num_heads, **kwargs)
        self.conv = nn.Conv2d(in_channels=value_dim, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, key, value, mask=None):
        print(f"Spatial: - Input shape to Spatial Attention: {value.shape}")

        # Ensure 'value' has 4 dimensions
        if value.dim() == 3:
            value = value.unsqueeze(1)  # Add a singleton channel dimension if missing

        if value.dim() != 4:
            raise ValueError(
                f"Spatial: - Expected 'value' tensor to have 4 dimensions (Batch, Channels, Height, Width), "
                f"but got {value.dim()} dimensions.")

        # Check if height and width are greater than 1
        batch_size, channels, height, width = value.size()
        if height <= 1 or width <= 1:
            print("Spatial Attention: - Skipping spatial attention as the height and width are <= 1.")
            return value, None  # Return original value and None for attention weights

        # Ensure 'value' has the correct number of channels for the convolution
        if value.size(1) != self.conv.in_channels:
            raise ValueError(f"Spatial: - Expected 'value' tensor to have {self.conv.in_channels} channels, "
                             f"but got {value.size(1)} channels.")

        # Apply the convolution to generate spatial attention map
        attn_map = self.conv(value)  # Shape: (Batch, 1, Height, Width)
        attn_map = self.sigmoid(attn_map)  # Shape: (Batch, 1, Height, Width)

        # Apply the attention map to 'value'
        output = value * attn_map  # Broadcasting to (Batch, Channels, Height, Width)

        return output, attn_map

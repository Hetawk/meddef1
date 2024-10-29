# attention.py

import torch
import torch.nn as nn

from .base.soft_attention import SoftAttention
from .base.self_attention import SelfAttention
from .base.local_attention import LocalAttention
from .base.hard_attention import HardAttention
from .base.global_attention import GlobalAttention
from .base.cross_attention import CrossAttention


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, attention_types=None, **kwargs):
        """
        Initializes Attention to handle multiple attention types.

        Parameters:
        - query_dim, key_dim, value_dim: Dimensions for queries, keys, and values.
        - num_heads: Number of attention heads.
        - attention_types: List of attention types to use, e.g., ['soft', 'self', 'local'].
        - **kwargs: Additional keyword arguments specific to each attention type.
        """
        super(Attention, self).__init__()
        self.attention_layers = nn.ModuleList()

        # Define available attention type mappings
        self.attention_map = {
            'soft': SoftAttention,
            'self': SelfAttention,
            'local': LocalAttention,
            'hard': HardAttention,
            'global': GlobalAttention,
            'cross': CrossAttention,
        }

        # Initialize the attention layers as per specified types
        if attention_types is None:
            raise ValueError("You must specify at least one attention type in `attention_types`.")

        for attn_type in attention_types:
            if attn_type not in self.attention_map:
                raise ValueError(f"Unsupported attention type: {attn_type}")
            self.attention_layers.append(
                self.attention_map[attn_type](query_dim, key_dim, value_dim, num_heads, **kwargs))

    def forward(self, query, key, value, mask=None):
        """
        Passes input through each of the selected attention layers in sequence.

        Parameters:
        - query, key, value: Input tensors.
        - mask: Optional mask for attention layers.

        Returns:
        - output: Combined output of all attention layers.
        - attn_weights: List of attention weights from each layer.
        """
        outputs = []
        attn_weights_list = []

        for attn_layer in self.attention_layers:
            output, attn_weights = attn_layer(query, key, value, mask)
            outputs.append(output)
            attn_weights_list.append(attn_weights)

        # Optionally, combine outputs (e.g., concatenate, add, average)
        combined_output = torch.cat(outputs, dim=-1)  # Concatenate along the feature dimension

        return combined_output, attn_weights_list

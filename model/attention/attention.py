import torch
import torch.nn as nn
from .base.soft_attention import SoftAttention
from .base.self_attention import SelfAttention
from .base.local_attention import LocalAttention
from .base.hard_attention import HardAttention
from .base.global_attention import GlobalAttention
from .base.cross_attention import CrossAttention
from .base.multi_head_attention import MultiHeadAttention
from .base.spatial_attention import SpatialAttention


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, attention_types=None, combine_method='concat', **kwargs):
        """
        Initializes Attention to handle multiple attention types with flexible combining methods.

        Parameters:
        - query_dim, key_dim, value_dim: Dimensions for queries, keys, and values.
        - num_heads: Number of attention heads.
        - attention_types: List of attention types to use, e.g., ['soft', 'self', 'local'].
        - combine_method: Method to combine outputs ('concat', 'average', 'add').
        - **kwargs: Additional keyword arguments specific to each attention type.
        """
        super(Attention, self).__init__()
        self.attention_layers = nn.ModuleList()
        self.combine_method = combine_method

        # Define available attention type mappings
        self.attention_map = {
            'soft': SoftAttention,
            'self': SelfAttention,
            'local': LocalAttention,
            'hard': HardAttention,
            'global': GlobalAttention,
            'cross': CrossAttention,
            'multi_head': MultiHeadAttention,
            'spatial': SpatialAttention
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
        Passes input through each of the selected attention layers and combines their outputs.

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

        # Combine outputs based on the specified method
        if self.combine_method == 'concat':
            combined_output = torch.cat(outputs, dim=-1)  # Concatenate along the feature dimension
        elif self.combine_method == 'average':
            combined_output = torch.mean(torch.stack(outputs), dim=0)  # Average the outputs
        elif self.combine_method == 'add':
            combined_output = sum(outputs)  # Element-wise addition
        else:
            raise ValueError(f"Unsupported combine method: {self.combine_method}")

        return combined_output, attn_weights_list

    def get_num_heads(self):
        """Retrieve the number of heads from the first attention layer."""
        if len(self.attention_layers) > 0:
            return self.attention_layers[0].num_heads
        else:
            raise ValueError("No attention layers initialized.")
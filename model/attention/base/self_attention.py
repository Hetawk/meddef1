import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention

class SelfAttention(BaseAttention):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1, **kwargs):
        super(SelfAttention, self).__init__(query_dim, key_dim, value_dim, num_heads)
        self.scale = (key_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Ensure the input tensors have the correct dimensions
        assert query.dim() == 3, f"Expected query to have 3 dimensions, got {query.dim()}"
        assert key.dim() == 3, f"Expected key to have 3 dimensions, got {key.dim()}"
        assert value.dim() == 3, f"Expected value to have 3 dimensions, got {value.dim()}"

        # # Print original shapes
        # print(f"SelfAttention: Original query shape: {query.shape}")
        # print(f"SelfAttention: Original key shape: {key.shape}")
        # print(f"SelfAttention: Original value shape: {value.shape}")

        # Linear transformations
        query_transformed = self.wq(query)  # Shape: (batch_size, seq_len, query_dim)
        key_transformed = self.wk(key)      # Shape: (batch_size, seq_len, key_dim)
        value_transformed = self.wv(value)  # Shape: (batch_size, seq_len, value_dim)

        # # Print transformed shapes
        # print(f"SelfAttention: Transformed query shape: {query_transformed.shape}")
        # print(f"SelfAttention: Transformed key shape: {key_transformed.shape}")
        # print(f"SelfAttention: Transformed value shape: {value_transformed.shape}")

        # Split heads
        query = self.split_heads(query_transformed, batch_size)  # Shape: (batch_size, num_heads, seq_len, depth)
        key = self.split_heads(key_transformed, batch_size)      # Shape: (batch_size, num_heads, seq_len, depth)
        value = self.split_heads(value_transformed, batch_size)  # Shape: (batch_size, num_heads, seq_len, depth)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

        # Apply mask if available
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(scores)  # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        output = torch.matmul(attn_weights, value)  # Shape: (batch_size, num_heads, seq_len_q, depth)

        output = self.combine_heads(output, batch_size)  # Shape: (batch_size, seq_len_q, num_heads * depth)
        output = self.dense(output)  # Final output

        return output, attn_weights

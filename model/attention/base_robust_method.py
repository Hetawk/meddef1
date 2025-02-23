# base_robust_method.py

import torch
import torch.nn as nn
from .attention import Attention  # Assuming Attention class is defined in model.attention


class BaseRobustMethod(nn.Module):
    def __init__(self, method_type, input_dim, output_dim, value_dim, **kwargs):
        super(BaseRobustMethod, self).__init__()
        self.method_type = method_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_dim = value_dim
        self.extra_params = kwargs  # Additional parameters for customization

        # Define mappings for robust method types
        self.method_map = {
            'attention': Attention,
            # Add other methods as necessary
        }

        if method_type not in self.method_map:
            raise ValueError(f"Unsupported method type: {method_type}")

        # Initialize the specific method based on `method_type`
        self.method = self.method_map[method_type](query_dim=input_dim, key_dim=input_dim, value_dim=value_dim,
                                                   **kwargs)

    def forward(self, query, key, value, mask=None):
        return self.method(query, key, value, mask)

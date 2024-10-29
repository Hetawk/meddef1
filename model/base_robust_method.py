# base_robust_method.py

import torch
import torch.nn as nn
from .attention.attention import Attention  # Import the Attention class

class BaseRobustMethod(nn.Module):
    def __init__(self, method_type, input_dim, output_dim, **kwargs):
        super(BaseRobustMethod, self).__init__()
        self.method_type = method_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define available robust method mappings
        self.method_map = {
            'attention': Attention,
            # Add other robust methods here
            # 'shift': ShiftOperation,
            # 'state_space': StateSpaceModel,
            # 'gnn': GraphNeuralNetwork,
            # 'rnn': RecurrentNeuralNetwork,
        }

        if method_type not in self.method_map:
            raise ValueError(f"Unsupported method type: {method_type}")

        self.method = self.method_map[method_type](input_dim, output_dim, **kwargs)

    def forward(self, x):
        return self.method(x)

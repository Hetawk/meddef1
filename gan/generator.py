# generator.py

import torch
import torch.nn as nn


class Generator(nn.Module):
    # output_dim should be 150528 for 3×224×224 images
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        # Define the generator network
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Forward pass through the generator
        return self.model(x)

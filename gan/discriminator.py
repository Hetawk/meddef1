# discriminator.py

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim):  # input_dim must equal 150528 for 3×224×224 flattened images
        super(Discriminator, self).__init__()
        # Define the discriminator network
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through the discriminator
        return self.model(x)

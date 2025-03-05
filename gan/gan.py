# gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from .generator import Generator
from .discriminator import Discriminator


class GAN:
    def __init__(self, noise_dim, data_dim, device):
        # data_dim should be 150528 for 3×224×224 images
        self.device = device
        # Initialize the generator and discriminator
        self.generator = Generator(noise_dim, data_dim).to(device)
        self.discriminator = Discriminator(data_dim).to(device)
        self.criterion = nn.BCELoss()
        # Optimizers for generator and discriminator
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def generate(self, noise):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(noise.to(self.device))


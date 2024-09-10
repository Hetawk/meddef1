# conditional_diffusion_model.py

import torch
import torch.nn as nn
import logging

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        logging.info(f"Initializing ConditionalDiffusionModel with {input_channels} input channels and {num_classes} classes.")
        super(ConditionalDiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * 32 * 32, num_classes)  # Adjusted for variable number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

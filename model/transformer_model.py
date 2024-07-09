# transformer_model.py

import torch
import torch.nn as nn
import logging

class TransformerModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, pretrained=False):
        logging.info(f"Initializing TransformerModel with {input_channels} input channels.")
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)
        self.fc = nn.Linear(512, num_classes)  # Adjusted for variable number of classes

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, 0, :])  # Aggregate across the sequence dimension
        return x

import torch
import torch.nn as nn


class MedicalFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_detector = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                       padding=1, groups=in_channels)
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5,
                      padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Add context aggregation
        self.context = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        edges = torch.tanh(self.edge_detector(x))
        texture = self.texture_analyzer(x)
        # Combine original features with edge and texture information
        combined = torch.cat([x, edges, texture], dim=1)
        return self.context(combined)

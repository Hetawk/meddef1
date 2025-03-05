import torch
import torch.nn as nn


class AdversarialFeatureDetector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.noise_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels),
            nn.Sigmoid()
        )
        self.feature_cleaner = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        noise_mask = self.noise_detector(x)
        cleaned_features = self.feature_cleaner(x * noise_mask)
        return cleaned_features + x  # Residual connection

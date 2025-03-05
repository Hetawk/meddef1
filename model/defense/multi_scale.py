import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_detector import AdversarialFeatureDetector
from .medical_features import MedicalFeatureExtractor


class MultiScaleFeatures(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ) for size in [(2, 2), (4, 4), (8, 8)]
        ])

        # Add feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1)
        )

    def forward(self, x):
        features = [x]
        for scale in self.scales:
            scaled = scale(x)
            features.append(F.interpolate(scaled, size=x.shape[-2:],
                                          mode='bilinear', align_corners=True))
        multi_scale = torch.cat(features, dim=1)
        return self.fusion(multi_scale)


class DefenseModule(nn.Module):
    """Complete defense module combining all mechanisms"""

    def __init__(self, in_channels):
        super().__init__()
        # Defense components
        self.adv_detector = AdversarialFeatureDetector(in_channels)
        self.medical_features = MedicalFeatureExtractor(in_channels)
        self.multi_scale = MultiScaleFeatures(in_channels)

        # Feature aggregation
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get features from each defense mechanism
        adv_features = self.adv_detector(x)
        med_features = self.medical_features(x)
        ms_features = self.multi_scale(x)

        # Combine features
        combined = torch.cat([adv_features, med_features, ms_features], dim=1)
        fused = self.feature_fusion(combined)

        # Apply channel attention
        attention = self.channel_attention(fused)
        out = fused * attention

        return out + x  # Residual connection

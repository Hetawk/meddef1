# meddef2.py

import torch
from torch import nn
from typing import Type, Tuple, Union, Optional

from model.attention.base_robust_method import BaseRobustMethod
from model.meddef.cbamsa import CBAMBottleneckBlock, CBAMBasicBlock
from model.meddef.rcbamsa import ResNetCBAMSA
from model.attention.base.self_attention import SelfAttention
from model.defense.multi_scale import DefenseModule


class MedDefBase(nn.Module):
    """
    The MedDefBase class initializes a single ResNetCBAMSA model with a given depth configuration.
    Combines CBAM backbone features with external SelfAttention.
    """

    def __init__(self, block: Union[Type[CBAMBasicBlock], Type[CBAMBottleneckBlock]],
                 layers: Tuple[int, int, int, int],
                 num_classes: int = 1000, input_channels: int = 3,
                 robust_method: Optional[BaseRobustMethod] = None):
        super(MedDefBase, self).__init__()

        # Initialize ResNetCBAMSA with the given block and layers
        self.resnet_cbamsa = ResNetCBAMSA(
            block, layers, num_classes=num_classes, input_channels=input_channels)

        # Add defense module
        channels = 512 * block.expansion
        self.defense = DefenseModule(channels)

        # Fix self-attention dimensions
        self.self_attention = SelfAttention(
            in_dim=channels,  # Input dimension
            key_dim=channels,  # Keep full dimension for key
            query_dim=channels,  # Keep full dimension for query
            value_dim=channels  # Keep full dimension for value
        )

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        # Add robust method
        self.robust_method = robust_method

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        # Get features from ResNet+CBAM+SA
        features = self.resnet_cbamsa(x)

        # Apply defense module
        defended = self.defense(features)

        # Process features properly
        B, C, H, W = defended.shape

        # Spatial features path
        spatial_features = defended  # [B, C, H, W]

        # Self-attention path with proper reshaping
        flat_features = defended.flatten(2)  # [B, C, HW]
        flat_features = flat_features.permute(0, 2, 1)  # [B, HW, C]

        # Apply self-attention and handle tuple return
        attended, _ = self.self_attention(
            query=flat_features,
            key=flat_features,
            value=flat_features
        )  # Now properly unpacking tuple

        # Reshape back to spatial form
        attended = attended.permute(0, 2, 1).view(B, C, H, W)

        # Combine features
        combined = torch.cat([spatial_features, attended], dim=1)
        fused = self.fusion(combined)

        # Apply robust method if available
        if self.robust_method is not None:
            fused, _ = self.robust_method(fused, fused, fused)

        # Final classification
        out = self.avgpool(fused)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

# Specific MedDef classes for each depth configuration remain unchanged


class MedDef2_0(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_0, self).__init__(CBAMBasicBlock,
                                        (2, 2, 2, 2), num_classes, input_channels, pretrained)


class MedDef2_1(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_1, self).__init__(CBAMBasicBlock,
                                        (3, 4, 6, 3), num_classes, input_channels, pretrained)


class MedDef2_2(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_2, self).__init__(CBAMBottleneckBlock,
                                        (3, 4, 6, 3), num_classes, input_channels, pretrained)


class MedDef2_3(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_3, self).__init__(CBAMBottleneckBlock,
                                        (3, 4, 23, 3), num_classes, input_channels, pretrained)


class MedDef2_4(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_4, self).__init__(CBAMBottleneckBlock,
                                        (3, 8, 36, 3), num_classes, input_channels, pretrained)


class MedDef2_5(nn.Module):
    """
    The MedDef2_5 class combines two ResNetCBAMSA models, each with depth 50.
    """

    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_5, self).__init__()

        # Initialize two ResNetCBAMSA models with depth 50
        self.resnet_cbamsa1 = ResNetCBAMSA(CBAMBottleneckBlock, (3, 4, 6, 3), num_classes=num_classes,
                                           input_channels=input_channels)
        self.resnet_cbamsa2 = ResNetCBAMSA(CBAMBottleneckBlock, (3, 4, 6, 3), num_classes=num_classes,
                                           input_channels=input_channels)

        # Remove the final fully connected layers from both models because we will use their feature outputs
        self.resnet_cbamsa1.fc = nn.Identity()
        self.resnet_cbamsa2.fc = nn.Identity()

        # Define a new fully connected layer that combines both ResNetCBAMSA feature outputs
        # Output size of ResNetCBAMSA's features
        resnet_cbamsa_out_features = 512 * CBAMBottleneckBlock.expansion
        combined_features = resnet_cbamsa_out_features * 2

        # Final fully connected layer
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        # Pass input through the first ResNetCBAMSA
        resnet_cbamsa1_features = self.resnet_cbamsa1(x)

        # Pass input through the second ResNetCBAMSA
        resnet_cbamsa2_features = self.resnet_cbamsa2(x)

        # Concatenate the features from both ResNetCBAMSA models
        combined_features = torch.cat(
            (resnet_cbamsa1_features, resnet_cbamsa2_features), dim=1)

        # Pass through the final fully connected layer
        output = self.fc(combined_features)
        return output


class MedDef2_6(nn.Module):
    """
    The MedDef2_6 class combines two ResNetCBAMSA models, each with depth 18.
    """

    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_6, self).__init__()

        # Initialize two ResNetCBAMSA models with depth 18
        self.resnet_cbamsa1 = ResNetCBAMSA(CBAMBasicBlock, (2, 2, 2, 2), num_classes=num_classes,
                                           input_channels=input_channels)
        self.resnet_cbamsa2 = ResNetCBAMSA(CBAMBasicBlock, (2, 2, 2, 2), num_classes=num_classes,
                                           input_channels=input_channels)

        # Remove the final fully connected layers from both models because we will use their feature outputs
        self.resnet_cbamsa1.fc = nn.Identity()
        self.resnet_cbamsa2.fc = nn.Identity()

        # Define a new fully connected layer that combines both ResNetCBAMSA feature outputs
        # Output size of ResNetCBAMSA's features
        resnet_cbamsa_out_features = 512 * CBAMBasicBlock.expansion
        combined_features = resnet_cbamsa_out_features * 2

        # Final fully connected layer
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        # Pass input through the first ResNetCBAMSA
        resnet_cbamsa1_features = self.resnet_cbamsa1(x)

        # Pass input through the second ResNetCBAMSA
        resnet_cbamsa2_features = self.resnet_cbamsa2(x)

        # Concatenate the features from both ResNetCBAMSA models
        combined_features = torch.cat(
            (resnet_cbamsa1_features, resnet_cbamsa2_features), dim=1)

        # Pass through the final fully connected layer
        output = self.fc(combined_features)
        return output


def get_meddef2(depth: float, input_channels: int = 3, num_classes: int = None,
                robust_method: Optional[BaseRobustMethod] = None) -> nn.Module:
    depth_to_block_layers = {
        2.0: (CBAMBasicBlock, (2, 2, 2, 2)),  # Depth 18
        2.1: (CBAMBasicBlock, (3, 4, 6, 3)),  # Depth 34
        2.2: (CBAMBottleneckBlock, (3, 4, 6, 3)),  # Depth 50
        2.3: (CBAMBottleneckBlock, (3, 4, 23, 3)),  # Depth 101
        2.4: (CBAMBottleneckBlock, (3, 8, 36, 3)),  # Depth 152
        2.5: MedDef2_5,  # Depth 50 x 2
        2.6: MedDef2_6,  # Depth 18 x 2
    }
    if depth not in depth_to_block_layers:
        raise ValueError(f"Unsupported meddef depth: {depth}")

    if depth in [2.5, 2.6]:
        return depth_to_block_layers[depth](num_classes=num_classes, input_channels=input_channels, pretrained=False)
    else:
        block, layers = depth_to_block_layers[depth]
        return MedDefBase(block, layers, num_classes=num_classes, input_channels=input_channels,
                          pretrained=False, robust_method=robust_method)

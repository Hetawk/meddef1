# meddef2.py

import torch
from torch import nn
from typing import Type, Tuple, Union, Optional

from model.attention.base_robust_method import BaseRobustMethod
from model.meddef.cbamsa import CBAMBottleneckBlock, CBAMBasicBlock
from model.meddef.rcbamsa import ResNetCBAMSA
from model.attention.base.self_attention import SelfAttention   # added import

class MedDefBase(nn.Module):
    """
    The MedDefBase class initializes a single ResNetCBAMSA model with a given depth configuration.
    Combines CBAM backbone features with external SelfAttention.
    """

    def __init__(self, block: Union[Type[CBAMBasicBlock], Type[CBAMBottleneckBlock]],
                 layers: Tuple[int, int, int, int],
                 num_classes: int, input_channels: int = 3, pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None):
        super(MedDefBase, self).__init__()

        # Initialize ResNetCBAMSA with the given block and layers
        self.resnet_cbamsa = ResNetCBAMSA(block, layers, num_classes=num_classes, input_channels=input_channels)

        # Remove the final fully connected layer because we will use the feature output
        self.resnet_cbamsa.fc = nn.Identity()

        # Final fully connected layer: input features dimension remains 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Define an external SelfAttention layer from the shared module
        self.self_attention = SelfAttention(512 * block.expansion,
                                            512 * block.expansion,
                                            512 * block.expansion)
        self.robust_method = robust_method

    def forward(self, x):
        # Extract features using ResNetCBAMSA backbone
        features = self.resnet_cbamsa(x)    # features shape: [batch, channels, H, W]

        # Flatten features
        x = features.view(features.size(0), -1)  # shape: [batch, channels_flat]

        # Apply self-attention: add sequence dimension and then remove it
        x_seq = x.unsqueeze(1)    # shape: [batch, 1, channels_flat]
        x_att, _ = self.self_attention(x_seq, x_seq, x_seq)
        x = x_att.squeeze(1)      # back to shape: [batch, channels_flat]

        if self.robust_method:
            x, _ = self.robust_method(x, x, x)

        # Final classification
        output = self.fc(x)
        return output

# Specific MedDef classes for each depth configuration remain unchanged
class MedDef2_0(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_0, self).__init__(CBAMBasicBlock, (2, 2, 2, 2), num_classes, input_channels, pretrained)

class MedDef2_1(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_1, self).__init__(CBAMBasicBlock, (3, 4, 6, 3), num_classes, input_channels, pretrained)

class MedDef2_2(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_2, self).__init__(CBAMBottleneckBlock, (3, 4, 6, 3), num_classes, input_channels, pretrained)

class MedDef2_3(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_3, self).__init__(CBAMBottleneckBlock, (3, 4, 23, 3), num_classes, input_channels, pretrained)

class MedDef2_4(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef2_4, self).__init__(CBAMBottleneckBlock, (3, 8, 36, 3), num_classes, input_channels, pretrained)

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
        resnet_cbamsa_out_features = 512 * CBAMBottleneckBlock.expansion  # Output size of ResNetCBAMSA's features
        combined_features = resnet_cbamsa_out_features * 2

        # Final fully connected layer
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        # Pass input through the first ResNetCBAMSA
        resnet_cbamsa1_features = self.resnet_cbamsa1(x)

        # Pass input through the second ResNetCBAMSA
        resnet_cbamsa2_features = self.resnet_cbamsa2(x)

        # Concatenate the features from both ResNetCBAMSA models
        combined_features = torch.cat((resnet_cbamsa1_features, resnet_cbamsa2_features), dim=1)

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
        resnet_cbamsa_out_features = 512 * CBAMBasicBlock.expansion  # Output size of ResNetCBAMSA's features
        combined_features = resnet_cbamsa_out_features * 2

        # Final fully connected layer
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        # Pass input through the first ResNetCBAMSA
        resnet_cbamsa1_features = self.resnet_cbamsa1(x)

        # Pass input through the second ResNetCBAMSA
        resnet_cbamsa2_features = self.resnet_cbamsa2(x)

        # Concatenate the features from both ResNetCBAMSA models
        combined_features = torch.cat((resnet_cbamsa1_features, resnet_cbamsa2_features), dim=1)

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

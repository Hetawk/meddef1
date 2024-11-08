# meddef.py

import torch
from torch import nn
from typing import Type, Tuple, Union, Optional

from model.attention.base_robust_method import BaseRobustMethod
from model.meddef.cbamsa import CBAMBottleneckBlock, CBAMBasicBlock
from model.meddef.rcbamsa import ResNetCBAMSA

class MedDefBase(nn.Module):
    """
    The MedDefBase class initializes a single ResNetCBAMSA model with a given depth configuration.
    """
    def __init__(self, block: Union[Type[CBAMBasicBlock], Type[CBAMBottleneckBlock]], layers: Tuple[int, int, int, int],
                 num_classes: int, input_channels: int = 3, pretrained: bool = False):
        super(MedDefBase, self).__init__()

        # Initialize ResNetCBAMSA with the given block and layers
        self.resnet_cbamsa = ResNetCBAMSA(block, layers, num_classes=num_classes, input_channels=input_channels)

        # Remove the final fully connected layer because we will use the feature output
        self.resnet_cbamsa.fc = nn.Identity()

        # Define a new fully connected layer that uses ResNetCBAMSA feature output
        resnet_cbamsa_out_features = 512 * block.expansion  # Output size of ResNetCBAMSA's features

        # Final fully connected layer
        self.fc = nn.Linear(resnet_cbamsa_out_features, num_classes)

    def forward(self, x):
        # Pass input through the ResNetCBAMSA
        resnet_cbamsa_features = self.resnet_cbamsa(x)

        # Pass through the final fully connected layer
        output = self.fc(resnet_cbamsa_features)
        return output

# Specific MedDef classes for each depth configuration
class MedDef1_0(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_0, self).__init__(CBAMBasicBlock, (2, 2, 2, 2), num_classes, input_channels, pretrained)

class MedDef1_1(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_1, self).__init__(CBAMBasicBlock, (3, 4, 6, 3), num_classes, input_channels, pretrained)

class MedDef1_2(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_2, self).__init__(CBAMBottleneckBlock, (3, 4, 6, 3), num_classes, input_channels, pretrained)

class MedDef1_3(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_3, self).__init__(CBAMBottleneckBlock, (3, 4, 23, 3), num_classes, input_channels, pretrained)

class MedDef1_4(MedDefBase):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_4, self).__init__(CBAMBottleneckBlock, (3, 8, 36, 3), num_classes, input_channels, pretrained)

class MedDef1_5(nn.Module):
    """
    The MedDef1_5 class combines two ResNetCBAMSA models, each with depth 50.
    """
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_5, self).__init__()

        # Initialize two ResNetCBAMSA models with depth 50
        self.resnet_cbamsa1 = ResNetCBAMSA(CBAMBottleneckBlock, (3, 4, 6, 3), num_classes=num_classes, input_channels=input_channels)
        self.resnet_cbamsa2 = ResNetCBAMSA(CBAMBottleneckBlock, (3, 4, 6, 3), num_classes=num_classes, input_channels=input_channels)

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

class MedDef1_6(nn.Module):
    """
    The MedDef1_6 class combines two ResNetCBAMSA models, each with depth 18.
    """
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef1_6, self).__init__()

        # Initialize two ResNetCBAMSA models with depth 18
        self.resnet_cbamsa1 = ResNetCBAMSA(CBAMBasicBlock, (2, 2, 2, 2), num_classes=num_classes, input_channels=input_channels)
        self.resnet_cbamsa2 = ResNetCBAMSA(CBAMBasicBlock, (2, 2, 2, 2), num_classes=num_classes, input_channels=input_channels)

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

def get_meddef(depth: float, input_channels: int = 3, num_classes: int = None, robust_method: Optional[BaseRobustMethod] = None) -> nn.Module:
    depth_to_block_layers = {
        1.0: (CBAMBasicBlock, (2, 2, 2, 2)), # Depth 18
        1.1: (CBAMBasicBlock, (3, 4, 6, 3)), # Depth 34
        1.2: (CBAMBottleneckBlock, (3, 4, 6, 3)), # Depth 50
        1.3: (CBAMBottleneckBlock, (3, 4, 23, 3)),  # Depth 101
        1.4: (CBAMBottleneckBlock, (3, 8, 36, 3)), # Depth 152
        1.5: MedDef1_5, # Depth 50 x 2
        1.6: MedDef1_6, # Depth 18 x 2
    }
    if depth not in depth_to_block_layers:
        raise ValueError(f"Unsupported meddef depth: {depth}")

    if depth in [1.5, 1.6]:
        return depth_to_block_layers[depth](num_classes=num_classes, input_channels=input_channels, pretrained=False)
    else:
        block, layers = depth_to_block_layers[depth]
        return MedDefBase(block, layers, num_classes=num_classes, input_channels=input_channels, pretrained=False)

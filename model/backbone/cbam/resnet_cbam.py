
import torch
import torch.nn as nn
from typing import Type, Tuple, Dict, Union, Optional
from model.attention.base_robust_method import BaseRobustMethod
from model.backbone.cbam.cbam_backbone import CBAMBasicBlock, CBAMBottleneckBlock

class ResNetModelWithCBAM(nn.Module):
    def __init__(self, block: Union[Type[CBAMBasicBlock], Type[CBAMBottleneckBlock]], layers: Tuple[int, int, int, int],
                 num_classes: int, input_channels: int = 3, robust_method: Optional[BaseRobustMethod] = None):
        super(ResNetModelWithCBAM, self).__init__()
        self.in_channels = 64
        self.block_expansion = block.expansion
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Adjust the CBAM layers to handle the block expansion
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)
        self.robust_method = robust_method

    def make_layer(self, block, out_channels, blocks, stride=1, bottleneck_channels=None):
        layers = []
        expansion = block.expansion

        # Handle the first block in the layer with stride if needed
        if block == CBAMBottleneckBlock:
            layers.append(block(self.in_channels, out_channels, stride=stride, bottleneck_channels=bottleneck_channels))
        else:
            layers.append(block(self.in_channels, out_channels, stride=stride))

        # Update in_channels for the following blocks in the layer
        self.in_channels = out_channels * expansion

        # Add the remaining blocks in this layer
        for _ in range(1, blocks):
            if block == CBAMBottleneckBlock:
                layers.append(block(self.in_channels, out_channels, bottleneck_channels=bottleneck_channels))
            else:
                layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels * expansion  # Ensures the next block starts with the correct in_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_without_fc(x)
        if self.robust_method:
            x, _ = self.robust_method(x, x, x)
            return x
        else:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def forward_without_fc(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


# Decorator to ensure num_classes is passed
def check_num_classes(func):
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)
    return wrapper

@check_num_classes
def get_resnet_with_cbam(depth: int, input_channels: int = 3, num_classes: int = None,
                         robust_method: Optional[BaseRobustMethod] = None) -> ResNetModelWithCBAM:
    depth_to_block_layers = {
        18: (CBAMBasicBlock, (2, 2, 2, 2)),
        34: (CBAMBasicBlock, (3, 4, 6, 3)),
        50: (CBAMBottleneckBlock, (3, 4, 6, 3)),
        101: (CBAMBottleneckBlock, (3, 4, 23, 3)),
        152: (CBAMBottleneckBlock, (3, 8, 36, 3)),
    }
    if depth not in depth_to_block_layers:
        raise ValueError(f"Unsupported cbam_resnet depth: {depth}")

    block, layers = depth_to_block_layers[depth]
    return ResNetModelWithCBAM(block, layers, num_classes=num_classes, input_channels=input_channels,
                               robust_method=robust_method)

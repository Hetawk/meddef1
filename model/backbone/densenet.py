import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, List, Optional
# Assuming this exists in your codebase
from model.attention.base_robust_method import BaseRobustMethod

# URL for pretrained DenseNet model weights
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

# Add new DenseLayer definition


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate,
                              kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv(self.relu(self.bn(x)))
        return new_features

# Modify DenseBlock to use DenseLayer and concatenate features


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(
                in_channels + i * growth_rate, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, 1))
            features.append(new_feat)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)


class DenseNetModel(nn.Module):
    def __init__(self, growth_rate: int, block_config: List[int], num_classes: int,
                 input_channels: int = 3, pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None,
                 arch_name: str = None):  # new parameter
        super(DenseNetModel, self).__init__()
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.arch_name = arch_name  # store architecture name for pretrained lookup

        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transition layers
        num_channels = 64
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_convs in enumerate(block_config):
            self.blocks.append(DenseBlock(
                num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(block_config) - 1:  # Don't add transition layer after the last block
                self.transitions.append(TransitionLayer(
                    num_channels, num_channels // 2))
                num_channels //= 2  # Halve the channels after transition layer

        # Final pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

        if pretrained:
            self.load_pretrained_weights()

        self.robust_method = robust_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for block, transition in zip(self.blocks, self.transitions):
            x = block(x)
            x = transition(x)

        x = self.blocks[-1](x)  # Ensure the last block is applied
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for classifier

        if self.robust_method:
            x, _ = self.robust_method(x, x, x)

        x = self.fc(x)
        return x

    def load_pretrained_weights(self):
        # Updated to use self.arch_name as key
        pretrained_dict = load_state_dict_from_url(
            model_urls[self.arch_name], progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items(
        ) if k in model_dict and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def get_densenet(depth: int, num_classes: int, pretrained: bool = False, input_channels: int = 3,
                 robust_method: Optional[BaseRobustMethod] = None) -> DenseNetModel:
    config = {
        121: ([6, 12, 24, 16], 32, 'densenet121'),
        169: ([6, 12, 32, 32], 32, 'densenet169'),
        201: ([6, 12, 48, 32], 32, 'densenet201'),
        161: ([6, 12, 36, 24], 48, 'densenet161')
    }
    if depth not in config:
        raise ValueError(f"Unsupported DenseNet depth: {depth}")

    block_config, growth_rate, model_name = config[depth]
    return DenseNetModel(growth_rate=growth_rate, block_config=block_config, num_classes=num_classes,
                         pretrained=pretrained, input_channels=input_channels, robust_method=robust_method,
                         arch_name=model_name)  # pass model_name as arch_name

import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url  # new
from typing import Optional, Tuple
import logging
from model.attention.base_robust_method import BaseRobustMethod  # new

# URL for pretrained ResNeXt model weights
model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_64x4d': 'https://download.pytorch.org/models/resnext101_64x4d-3b2fe3d8.pth',
}


class ResidualBlock(nn.Module):
    expansion = 1
    logged = False

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if not self.logged and torch.cuda.current_device() == 0:  # Only log for the first GPU to reduce clutter
            logging.info("Forward pass in ResidualBlock.")
            self.logged = True  # Set this to True after the first log
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXtBlock(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, cardinality, stride=1):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(in_channels, out_channels *
                               self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * self.expansion)
        self.conv2 = nn.Conv2d(out_channels * self.expansion, out_channels * self.expansion,
                               kernel_size=3, stride=stride, padding=1, groups=self.cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.conv3 = nn.Conv2d(out_channels * self.expansion,
                               out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXtModel(nn.Module):
    def __init__(self, layers: Tuple[int, int, int, int], cardinality: int, num_classes: int,
                 input_channels: int = 3, pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None):
        super(ResNeXtModel, self).__init__()
        self.in_channels = 64
        self.block_expansion = 2
        self.layers_cfg = layers  # store layers configuration
        self.robust_method = robust_method
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(cardinality, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(cardinality, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(cardinality, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(cardinality, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)

        if pretrained:
            self.load_pretrained_weights(layers, num_classes, input_channels)

    def make_layer(self, cardinality: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(ResNeXtBlock(self.in_channels,
                      out_channels, cardinality, stride))
        self.in_channels = out_channels * self.block_expansion
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock(
                self.in_channels, out_channels, cardinality))
        return nn.Sequential(*layers)

    def forward_without_fc(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # shape: (batch, channels, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_without_fc(x)
        if self.robust_method:
            x, _ = self.robust_method(x, x, x)
            return x  # Maintain 4D shape for attention layers if needed
        else:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def load_pretrained_weights(self, layers: Tuple[int, int, int, int], num_classes: int, input_channels: int):
        if sum(layers) == 16:  # ResNeXt50
            arch_key = 'resnext50_32x4d'
        elif sum(layers) == 30:  # ResNeXt101_32x8d
            arch_key = 'resnext101_32x8d'
        elif sum(layers) == 32:  # ResNeXt101_64x4d
            arch_key = 'resnext101_64x4d'
        else:
            raise ValueError(
                "Unsupported architecture for loading pretrained weights.")

        pretrained_dict = load_state_dict_from_url(
            model_urls[arch_key], progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        if input_channels != 3:
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)


def ResNeXt50(cardinality=32, num_classes=None, input_channels=3, pretrained=False, robust_method=None):
    return ResNeXtModel((3, 4, 6, 3), cardinality,
                        num_classes=num_classes, input_channels=input_channels,
                        pretrained=pretrained, robust_method=robust_method)


def ResNeXt101_32x8d(cardinality=32, num_classes=None, input_channels=3, pretrained=False, robust_method=None):
    return ResNeXtModel((3, 4, 23, 3), cardinality,
                        num_classes=num_classes, input_channels=input_channels,
                        pretrained=pretrained, robust_method=robust_method)


def ResNeXt101_64x4d(cardinality=64, num_classes=None, input_channels=3, pretrained=False, robust_method=None):
    return ResNeXtModel((3, 4, 23, 3), cardinality,
                        num_classes=num_classes, input_channels=input_channels,
                        pretrained=pretrained, robust_method=robust_method)

# Add check_num_classes decorator and get_resnext function


def check_num_classes(func):
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)
    return wrapper


@check_num_classes
def get_resnext(depth: int, pretrained: bool = False, input_channels: int = 3, num_classes: int = None,
                robust_method: Optional[BaseRobustMethod] = None) -> ResNeXtModel:
    # Mapping: key is depth (or a special key for ResNeXt101_64x4d)
    # Here, depth 50 returns ResNeXt50, depth 101 returns ResNeXt101_32x8d,
    # and depth 10164 returns ResNeXt101_64x4d (use 10164 as alias).
    depth_to_config = {
        50: (32, (3, 4, 6, 3), 'resnext50_32x4d'),
        101: (32, (3, 4, 23, 3), 'resnext101_32x8d'),
        10164: (64, (3, 4, 23, 3), 'resnext101_64x4d'),
    }
    if depth not in depth_to_config:
        raise ValueError(f"Unsupported ResNeXt depth: {depth}")
    cardinality, layers, arch_key = depth_to_config[depth]
    # Instantiate ResNeXtModel; load_pretrained_weights will use the arch_key via model_urls mapping.
    return ResNeXtModel(layers, cardinality, num_classes=num_classes, input_channels=input_channels,
                        pretrained=pretrained, robust_method=robust_method)

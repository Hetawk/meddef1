import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Tuple, Dict, Union, Optional
from model.attention.base_robust_method import BaseRobustMethod

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # Corrected this line to use the output of conv1
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetModel(nn.Module):
    def __init__(self, block: Union[Type[BasicBlock], Type[Bottleneck]], layers: Tuple[int, int, int, int],
                 num_classes: int, input_channels: int = 3, pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None):
        super(ResNetModel, self).__init__()
        self.in_channels = 64
        self.block_expansion = block.expansion
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)

        if pretrained:
            self.load_pretrained_weights(block, layers, num_classes, input_channels)

        self.robust_method = robust_method  # Initialize the robust method module

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_without_fc(x)
        if self.robust_method:
            x, _ = self.robust_method(x, x, x)  # Apply robust method
            return x  # Maintain 4D shape, required for attention layers in MSARNet
        else:
            x = x.view(x.size(0), -1)  # Flatten to 2D [batch_size, channels]
            # print(f"ResNet: - Shape before fc layer: {x.shape}")
            x = self.fc(x)
            # print(f"ResNet: - Shape after fc layer: {x.shape}")
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
        x = self.avgpool(x)  # Output will be 4D with shape [batch_size, channels, 1, 1]
        return x

    def load_pretrained_weights(self, block: Union[Type[BasicBlock], Type[Bottleneck]],
                                layers: Tuple[int, int, int, int], num_classes: int, input_channels: int):
        depth_to_url: Dict[Union[Tuple[Type[BasicBlock], Tuple[int, int, int, int]], Tuple[
            Type[Bottleneck], Tuple[int, int, int, int]]], str] = {
            (BasicBlock, (2, 2, 2, 2)): model_urls['resnet18'],
            (BasicBlock, (3, 4, 6, 3)): model_urls['resnet34'],
            (Bottleneck, (3, 4, 6, 3)): model_urls['resnet50'],
            (Bottleneck, (3, 4, 23, 3)): model_urls['resnet101'],
            (Bottleneck, (3, 8, 36, 3)): model_urls['resnet152'],
        }
        url = depth_to_url.get((block, layers))
        if url is None:
            raise ValueError("No pretrained model available for the specified architecture.")

        pretrained_dict = load_state_dict_from_url(url, progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fc = nn.Linear(512 * self.block_expansion, num_classes)

def check_num_classes(func):
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)

    return wrapper

@check_num_classes
def get_resnet(depth: int, pretrained: bool = False, input_channels: int = 3, num_classes: int = None,
               robust_method: Optional[BaseRobustMethod] = None) -> ResNetModel:
    depth_to_block_layers = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }
    if depth not in depth_to_block_layers:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    block, layers = depth_to_block_layers[depth]
    return ResNetModel(block, layers, num_classes=num_classes, pretrained=pretrained, input_channels=input_channels,
                       robust_method=robust_method)

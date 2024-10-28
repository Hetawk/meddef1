import torch
import torch.nn as nn
import torchvision.models as models
import logging
from urllib.request import urlopen
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
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

    def forward(self, x):
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

    def __init__(self, in_channels, out_channels, stride=1):
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

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
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
    def __init__(self, block, layers, num_classes, input_channels=3, pretrained=False):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def load_pretrained_weights(self, block, layers, num_classes, input_channels):
        if block == BasicBlock and layers == [2, 2, 2, 2]:
            url = model_urls['resnet18']
        elif block == BasicBlock and layers == [3, 4, 6, 3]:
            url = model_urls['resnet34']
        elif block == Bottleneck and layers == [3, 4, 6, 3]:
            url = model_urls['resnet50']
        elif block == Bottleneck and layers == [3, 4, 23, 3]:
            url = model_urls['resnet101']
        elif block == Bottleneck and layers == [3, 8, 36, 3]:
            url = model_urls['resnet152']
        else:
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
def ResNet18(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

@check_num_classes
def ResNet34(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

@check_num_classes
def ResNet50(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

@check_num_classes
def ResNet101(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

@check_num_classes
def ResNet152(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

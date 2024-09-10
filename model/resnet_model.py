# resnet_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import logging

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
        # Check if the torchvision version supports the new weights attribute
        if hasattr(models, 'ResNet18_Weights'):
            if block == BasicBlock and layers == [2, 2, 2, 2]:
                pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif block == BasicBlock and layers == [3, 4, 6, 3]:
                pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif block == Bottleneck and layers == [3, 4, 6, 3]:
                pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            elif block == Bottleneck and layers == [3, 4, 23, 3]:
                pretrained_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            elif block == Bottleneck and layers == [3, 8, 36, 3]:
                pretrained_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            else:
                raise ValueError("No pretrained model available for the specified architecture.")
        else:
            if block == BasicBlock and layers == [2, 2, 2, 2]:
                pretrained_model = models.resnet18(pretrained=True)
            elif block == BasicBlock and layers == [3, 4, 6, 3]:
                pretrained_model = models.resnet34(pretrained=True)
            elif block == Bottleneck and layers == [3, 4, 6, 3]:
                pretrained_model = models.resnet50(pretrained=True)
            elif block == Bottleneck and layers == [3, 4, 23, 3]:
                pretrained_model = models.resnet101(pretrained=True)
            elif block == Bottleneck and layers == [3, 8, 36, 3]:
                pretrained_model = models.resnet152(pretrained=True)
            else:
                raise ValueError("No pretrained model available for the specified architecture.")

        # Load pretrained weights, excluding the final layer
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # Adjust the first convolutional layer if the input channels are not 3
        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final layer with a new one that matches the number of classes for your dataset
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)


def ResNet18(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)


def ResNet34(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)


def ResNet50(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)


def ResNet101(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)


def ResNet152(pretrained=False, input_channels=3, num_classes=None):
    return ResNetModel(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

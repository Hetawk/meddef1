import torch
import torch.nn as nn
import torchvision.models as models
import logging


class ResidualBlock(nn.Module):
    expansion = 1
    logged = False

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
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
        self.conv1 = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * self.expansion)
        self.conv2 = nn.Conv2d(out_channels * self.expansion, out_channels * self.expansion,
                               kernel_size=3, stride=stride, padding=1, groups=self.cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.conv3 = nn.Conv2d(out_channels * self.expansion, out_channels * self.expansion, kernel_size=1, bias=False)
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


class ResNeXtModel(nn.Module):
    def __init__(self, block, layers, cardinality, num_classes, input_channels=3, pretrained=False):
        super(ResNeXtModel, self).__init__()
        self.in_channels = 64
        self.block_expansion = block.expansion
        self.layers = layers  # Add this line to store the layers structure
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], cardinality, stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], cardinality, stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], cardinality, stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)

        if pretrained:
            self.load_pretrained_weights(input_channels, num_classes)

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

    def make_layer(self, block, out_channels, blocks, cardinality, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, cardinality, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, cardinality))
        return nn.Sequential(*layers)

    def load_pretrained_weights(self, input_channels, num_classes):
        # Correctly choose the pretrained model based on the architecture
        if self.block_expansion == 2 and sum(self.layers) == 16:  # ResNeXt50
            pretrained_model = models.resnext50_32x4d(pretrained=True)
        elif self.block_expansion == 2 and sum(self.layers) == 30:  # ResNeXt101 32x8d
            pretrained_model = models.resnext101_32x8d(pretrained=True)
        elif self.block_expansion == 2 and sum(self.layers) == 32:  # ResNeXt101 64x4d
            pretrained_model = models.resnext101_64x4d(pretrained=True)
        else:
            raise ValueError("Unsupported architecture for loading pretrained weights.")

        # Load pretrained weights, excluding the final layer
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # Adjust the first convolutional layer if the input channels are not 3
        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Replace the final layer with a new one that matches the number of classes for your dataset
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)


def ResNeXt50(cardinality=32, num_classes=None, input_channels=3, pretrained=False):
    return ResNeXtModel(ResNeXtBlock, [3, 4, 6, 3], cardinality, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def ResNeXt101_32x8d(cardinality=32, num_classes=None, input_channels=3, pretrained=False):
    return ResNeXtModel(ResNeXtBlock, [3, 4, 23, 3], cardinality, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def ResNeXt101_64x4d(cardinality=64, num_classes=None, input_channels=3, pretrained=False):
    return ResNeXtModel(ResNeXtBlock, [3, 4, 23, 3], cardinality, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)
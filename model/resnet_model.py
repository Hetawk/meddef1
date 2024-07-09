import torch
import torch.nn as nn
import torchvision.models as models
import logging


class ResidualBlock(nn.Module):
    expansion = 1  # Adding the expansion attribute
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


class ResNetModel(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_channels=3, pretrained=False):
        super(ResNetModel, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if pretrained:
            self.load_pretrained_weights(input_channels)

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
            layers.append(block(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def load_pretrained_weights(self, input_channels):
        # Load pretrained weights from torchvision models
        pretrained_model = models.resnet18(pretrained=True)

        if input_channels == 3:
            self.conv1.load_state_dict(pretrained_model.conv1.state_dict())
        else:
            # For non-standard input channels, initialize the weights of the first conv layer
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1.load_state_dict(pretrained_model.bn1.state_dict())
        self.layer1.load_state_dict(pretrained_model.layer1.state_dict())
        self.layer2.load_state_dict(pretrained_model.layer2.state_dict())
        self.layer3.load_state_dict(pretrained_model.layer3.state_dict())
        self.layer4.load_state_dict(pretrained_model.layer4.state_dict())
        self.fc.load_state_dict(pretrained_model.fc.state_dict())


def ResNet18(pretrained=False, input_channels=3):
    return ResNetModel(ResidualBlock, [2, 2, 2, 2], pretrained=pretrained, input_channels=input_channels)


def ResNet34(pretrained=False, input_channels=3):
    return ResNetModel(ResidualBlock, [3, 4, 6, 3], pretrained=pretrained, input_channels=input_channels)


def ResNet50(pretrained=False, input_channels=3):
    return ResNetModel(ResidualBlock, [3, 4, 6, 3], pretrained=pretrained, input_channels=input_channels)


def ResNet101(pretrained=False, input_channels=3):
    return ResNetModel(ResidualBlock, [3, 4, 23, 3], pretrained=pretrained, input_channels=input_channels)


def ResNet152(pretrained=False, input_channels=3):
    return ResNetModel(ResidualBlock, [3, 8, 36, 3], pretrained=pretrained, input_channels=input_channels)

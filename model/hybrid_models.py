import torch
import torch.nn as nn
import torchvision.models as models
import logging


class ResidualBlock(nn.Module):
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
        logging.info("Forward pass in ResidualBlock.")
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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.make_conv(in_channels + i * growth_rate, growth_rate))

    def make_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        logging.info("Forward pass in DenseBlock.")
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class HybridModel(nn.Module):
    def __init__(self, num_blocks_resnet, num_blocks_densenet, growth_rate=12, reduction=0.5, num_classes=10,
                 pretrained_resnet=False, pretrained_densenet=False):
        super(HybridModel, self).__init__()
        self.resnet = self.create_resnet(num_blocks_resnet, pretrained=pretrained_resnet)
        self.densenet = self.create_densenet(num_blocks_densenet, growth_rate, reduction,
                                             pretrained=pretrained_densenet)
        self.fc = nn.Linear(512 + 1024, num_classes)  # Adjust based on the output sizes of ResNet and DenseNet

    def create_resnet(self, num_blocks, pretrained=False):
        resnet = models.resnet18(pretrained=pretrained)
        # Modify layers if needed
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def create_densenet(self, num_blocks, growth_rate, reduction, pretrained=False):
        densenet = models.densenet121(pretrained=pretrained)
        # Modify layers if needed
        return nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1,
            densenet.features.transition1,
            densenet.features.denseblock2,
            densenet.features.transition2,
            densenet.features.denseblock3,
            densenet.features.transition3,
            densenet.features.denseblock4,
            densenet.features.norm5
        )

    def forward(self, x):
        logging.info("Forward pass in HybridModel.")
        resnet_features = self.resnet(x)
        densenet_features = self.densenet(x)

        # Flatten and concatenate features from ResNet and DenseNet
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        densenet_features = densenet_features.view(densenet_features.size(0), -1)
        combined_features = torch.cat((resnet_features, densenet_features), dim=1)

        # Fully connected layer
        output = self.fc(combined_features)
        return output


def HybridResNetDenseNet(num_blocks_resnet, num_blocks_densenet, growth_rate=12, reduction=0.5, num_classes=10,
                         pretrained_resnet=False, pretrained_densenet=False):
    return HybridModel(num_blocks_resnet, num_blocks_densenet, growth_rate, reduction, num_classes, pretrained_resnet,
                       pretrained_densenet)

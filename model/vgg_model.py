import torch
import torch.nn as nn
import torchvision.models as models
import logging


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if not hasattr(self, 'logged') or not self.logged and torch.cuda.current_device() == 0:
            logging.info("Forward pass in VGGBlock.")
            self.logged = True
        x = self.conv(x)
        x = self.pool(x)
        return x


class VGGModel(nn.Module):
    def __init__(self, config, num_classes=1000, input_channels=3, pretrained=False):
        super(VGGModel, self).__init__()
        self.features = self.make_layers(config, input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if pretrained:
            self.load_pretrained_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, config, input_channels):
        layers = []
        in_channels = input_channels
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def load_pretrained_weights(self):
        pretrained_model = models.vgg16(pretrained=True)
        self.load_state_dict(pretrained_model.state_dict())


def VGG11(pretrained=False, input_channels=3, num_classes=1000):
    config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def VGG13(pretrained=False, input_channels=3, num_classes=1000):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def VGG16(pretrained=False, input_channels=3, num_classes=1000):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def VGG19(pretrained=False, input_channels=3, num_classes=1000):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)

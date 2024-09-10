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
    def __init__(self, config, num_classes, input_channels=3, pretrained=False):
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
            self.load_pretrained_weights(input_channels, num_classes)

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

    def load_pretrained_weights(self, input_channels, num_classes):
        # Load the VGG16 model from torchvision and adjust for input channels and num_classes
        pretrained_model = models.vgg16(pretrained=True)

        if input_channels != 3:
            # Handle non-standard input channels
            self.features[0].weight.data = pretrained_model.features[0].weight.data.mean(dim=1,
                                                                                         keepdim=True)  # Average RGB channels

        # Load pretrained weights into the feature extractor
        feature_dict = {k: v for k, v in pretrained_model.features.state_dict().items()}
        self.features.load_state_dict(feature_dict, strict=False)

        # Handle classifier adjustment
        in_features = 512 * 7 * 7
        self.classifier[0] = nn.Linear(in_features, 4096)
        self.classifier[3] = nn.Linear(4096, 4096)
        self.classifier[6] = nn.Linear(4096, num_classes)

        # Optionally: log successful loading
        if torch.cuda.is_available():
            logging.info(
                f"Pretrained weights loaded with input_channels={input_channels} and num_classes={num_classes} onto GPU.")
        else:
            logging.info(
                f"Pretrained weights loaded with input_channels={input_channels} and num_classes={num_classes} onto CPU.")


def VGG11(num_classes, input_channels=3, pretrained=False):
    config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def VGG13(num_classes, input_channels=3, pretrained=False):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def VGG16(num_classes, input_channels=3, pretrained=False):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)


def VGG19(num_classes, input_channels=3, pretrained=False):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGGModel(config, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)

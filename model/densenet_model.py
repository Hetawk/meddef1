
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            self.make_layer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])

    def make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.layer(x)


class DenseNetModel(nn.Module):
    def __init__(self, growth_rate, num_blocks, num_classes, input_channels=3, reduction=0.5, pretrained=False,
                 model_name=None):
        super(DenseNetModel, self).__init__()
        self.growth_rate = growth_rate

        # Initial layers
        num_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(input_channels, num_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transition layers
        self.blocks, self.transitions = nn.ModuleList(), nn.ModuleList()
        for i, layers in enumerate(num_blocks):
            self.blocks.append(DenseBlock(num_features, growth_rate, layers))
            num_features += layers * growth_rate
            if i < len(num_blocks) - 1:
                out_features = int(num_features * reduction)
                self.transitions.append(TransitionLayer(num_features, out_features))
                num_features = out_features

        # Final layers
        self.bn_final = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

        if pretrained:
            self.load_pretrained_weights(input_channels, num_classes, model_name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block, transition in zip(self.blocks, self.transitions):
            x = block(x)
            x = transition(x)

        x = self.blocks[-1](x)  # Ensure the last block is applied
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_pretrained_weights(self, input_channels, num_classes, model_name):
        if model_name not in model_urls:
            raise ValueError(f"No pretrained model available for {model_name}")

        pretrained_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 2 * self.growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(self.fc.in_features, num_classes)


def get_densenet(depth, pretrained=False, input_channels=3, num_classes=None):
    config = {
        121: ([6, 12, 24, 16], 32, 'densenet121'),
        169: ([6, 12, 32, 32], 32, 'densenet169'),
        201: ([6, 12, 48, 32], 32, 'densenet201'),
        161: ([6, 12, 36, 24], 48, 'densenet161')
    }
    if depth not in config:
        raise ValueError(f"Unsupported DenseNet depth: {depth}")

    num_blocks, growth_rate, model_name = config[depth]
    return DenseNetModel(growth_rate, num_blocks, num_classes, input_channels, pretrained=pretrained,
                         model_name=model_name)
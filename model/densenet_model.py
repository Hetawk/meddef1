import torch
import torch.nn as nn
import logging
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
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        output = torch.cat(features, 1)
        return output

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))  # Use AdaptiveAvgPool2d instead of AvgPool2d
        )

    def forward(self, x):
        output = self.transition(x)
        return output

class DenseNetModel(nn.Module):
    logged = False

    def __init__(self, num_blocks, num_classes, input_channels=3, growth_rate=12, reduction=0.5, pretrained=False, model_name=None):
        super(DenseNetModel, self).__init__()
        self.growth_rate = growth_rate

        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transition layers
        num_features = 2 * growth_rate
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(num_features, growth_rate, num_layers))
            num_features += num_layers * growth_rate
            if i != len(num_blocks) - 1:
                out_features = int(num_features * reduction)
                self.transition_layers.append(TransitionLayer(num_features, out_features))
                num_features = out_features

        # Final layers
        self.bn_final = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Use AdaptiveAvgPool2d
        self.fc = nn.Linear(num_features, num_classes)

        if pretrained:
            self.load_pretrained_weights(input_channels, num_classes, model_name)

    def forward(self, x):
        if not self.logged and torch.cuda.current_device() == 0:  # Only log for the first GPU to reduce clutter
            logging.info("Forward pass in DenseNetModel.")
            self.logged = True  # Set this to True after the first log
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.bn_final(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_pretrained_weights(self, input_channels, num_classes, model_name):
        if model_name not in model_urls:
            raise ValueError(f"No pretrained model available for {model_name}")

        # Load pretrained weights from the URL
        pretrained_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.fc = nn.Linear(self.fc.in_features, num_classes)

def DenseNet121(pretrained=False, input_channels=3, num_classes=None):
    return DenseNetModel([6, 12, 24, 16], growth_rate=32, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained, model_name='densenet121')

def DenseNet169(pretrained=False, input_channels=3, num_classes=None):
    return DenseNetModel([6, 12, 32, 32], growth_rate=32, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained, model_name='densenet169')

def DenseNet201(pretrained=False, input_channels=3, num_classes=None):
    return DenseNetModel([6, 12, 48, 32], growth_rate=32, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained, model_name='densenet201')

def DenseNet264(pretrained=False, input_channels=3, num_classes=None):
    return DenseNetModel([6, 12, 64, 48], growth_rate=32, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained, model_name='densenet161')
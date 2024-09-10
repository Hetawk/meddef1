import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
from torchvision.models import mobilenet_v2, mobilenet_v3_small


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, input_channels=3):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(input_channels, input_channel, stride=2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.classifier(x)
        return x

    def load_pretrained_weights(self, input_channels):
        if input_channels == 3:
            pretrained_model = mobilenet_v2(pretrained=True)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def MobileNetV2Model(pretrained=False, input_channels=3, num_classes=None):
    model = MobileNetV2(num_classes=num_classes, input_channels=input_channels)
    if pretrained:
        model.load_pretrained_weights(input_channels)
    return model


class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3) / 6


class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_planes, inplanes, 1),
            hsigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        return x * self.reduce_expand(x_se)


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            hswish()
        )


class InvertedResidualConfig:
    def __init__(self, inplanes, outplanes, kernel_size, stride, expand_ratio, se_planes):
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_planes = se_planes


class MobileNetV3(nn.Module):
    def __init__(self, num_classes, cfgs=None, input_channels=3):
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs
        self.conv1 = ConvBNActivation(input_channels, 16, 3, 2)
        self.layers = self._make_layers()
        self.conv2 = ConvBNActivation(cfgs[-1].inplanes, 960, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(960, num_classes)

    def _make_layers(self):
        layers = []
        for cfg in self.cfgs:
            layers.append(InvertedResidual(cfg))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def load_pretrained_weights(self, input_channels):
        if input_channels == 3:
            pretrained_model = mobilenet_v3_small(pretrained=True)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def MobileNetV3SmallModel(pretrained=False, input_channels=3, num_classes=None):
    cfgs = [
        InvertedResidualConfig(16, 16, 3, 2, 1, 4),
        InvertedResidualConfig(16, 24, 3, 2, 2, 3),
        InvertedResidualConfig(24, 24, 3, 1, 2.5, 3),
        InvertedResidualConfig(24, 40, 5, 2, 2.5, 3),
        InvertedResidualConfig(40, 40, 5, 1, 2.5, 3),
        InvertedResidualConfig(40, 40, 5, 1, 2.5, 3),
        InvertedResidualConfig(40, 48, 5, 1, 2.5, 3),
        InvertedResidualConfig(48, 48, 5, 1, 2.5, 3),
        InvertedResidualConfig(48, 96, 5, 2, 2.5, 3),
        InvertedResidualConfig(96, 96, 5, 1, 2.5, 3),
    ]
    model = MobileNetV3(cfgs=cfgs, input_channels=input_channels, num_classes=num_classes)
    if pretrained:
        model.load_pretrained_weights(input_channels)
    return model

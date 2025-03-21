import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Union, Type
from torchvision.models import mobilenet_v2, mobilenet_v3_small
from model.attention.base_robust_method import BaseRobustMethod


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, groups=groups, bias=False),
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
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
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
    def __init__(self, num_classes, width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8, input_channels=3, robust_method=None):
        super(MobileNetV2, self).__init__()
        self.robust_method = robust_method
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
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(
            input_channel, self.last_channel, kernel_size=1))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    def forward_without_fc(self, x):
        """Extract features before classification head"""
        x = self.features(x)
        x = self.avgpool(x)  # Returns 4D tensor [B, C, 1, 1]
        return x

    def forward(self, x):
        x = self.forward_without_fc(x)

        if self.robust_method:
            # Apply robust method if available
            x, _ = self.robust_method(x, x, x)
            return x  # Return 4D tensor for compatibility with attention modules
        else:
            # Standard forward path with classification
            x = torch.flatten(x, 1)  # Flatten to [B, C]
            x = self.classifier(x)
            return x

    def load_pretrained_weights(self, input_channels):
        """Load pretrained weights from torchvision model"""
        if input_channels == 3:  # Only load if standard RGB input
            logging.info("Loading pretrained MobileNetV2 weights")
            pretrained_model = mobilenet_v2(pretrained=True)
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_model.state_dict().items()
                if k in model_dict and 'classifier' not in k
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            logging.info(
                f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
        else:
            logging.info(
                f"Skipping pretrained weights: input has {input_channels} channels (not RGB)")


# MobileNetV3 implementation

class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6


class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileNetV3Block(nn.Module):
    """Mobile Inverted Residual Bottleneck Block for MobileNetV3"""

    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3Block, self).__init__()
        self.identity = stride == 1 and inp == oup

        activation = hswish() if use_hs else nn.ReLU(inplace=True)

        layers = []
        # Expand
        if hidden_dim != inp:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation
        ])

        # SE
        if use_se:
            layers.append(SEModule(hidden_dim))

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small implementation"""

    def __init__(self, num_classes=1000, input_channels=3, robust_method=None):
        super(MobileNetV3Small, self).__init__()
        self.robust_method = robust_method

        # Configuration for MobileNetV3-Small
        # [in_channels, exp_size, out_channels, kernel_size, stride, use_SE, use_HS]
        cfg = [
            [16, 16, 16, 3, 2, True, False],     # 0
            [16, 72, 24, 3, 2, False, False],    # 1
            [24, 88, 24, 3, 1, False, False],    # 2
            [24, 96, 40, 5, 2, True, True],      # 3
            [40, 240, 40, 5, 1, True, True],     # 4
            [40, 240, 40, 5, 1, True, True],     # 5
            [40, 120, 48, 5, 1, True, True],     # 6
            [48, 144, 48, 5, 1, True, True],     # 7
            [48, 288, 96, 5, 2, True, True],     # 8
            [96, 576, 96, 5, 1, True, True],     # 9
            [96, 576, 96, 5, 1, True, True],     # 10
        ]

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )

        # Building blocks
        for idx, (inp, exp, oup, kernel, stride, use_se, use_hs) in enumerate(cfg):
            self.features.add_module(f'block{idx}',
                                     MobileNetV3Block(inp, oup, exp, kernel, stride, use_se, use_hs))

        # Final layers
        self.features.add_module('conv_last', nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            hswish()
        ))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(576, 1280),
            hswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward_without_fc(self, x):
        """Extract features before classification head"""
        x = self.features(x)
        x = self.avgpool(x)  # Returns 4D tensor [B, C, 1, 1]
        return x

    def forward(self, x):
        x = self.forward_without_fc(x)

        if self.robust_method:
            # Apply robust method if available
            x, _ = self.robust_method(x, x, x)
            return x  # Return 4D tensor for compatibility with attention modules
        else:
            # Standard forward path with classification
            x = torch.flatten(x, 1)  # Flatten to [B, C]
            x = self.classifier(x)
            return x

    def load_pretrained_weights(self, input_channels):
        """Load pretrained weights from torchvision model"""
        if input_channels == 3:  # Only load if standard RGB input
            try:
                logging.info("Loading pretrained MobileNetV3-Small weights")
                pretrained_model = mobilenet_v3_small(pretrained=True)
                model_dict = self.state_dict()

                # Filter out classifier weights
                pretrained_dict = {
                    k: v for k, v in pretrained_model.state_dict().items()
                    if k in model_dict and 'classifier' not in k
                }

                # Update model weights
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
                logging.info(
                    f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
            except Exception as e:
                logging.error(f"Failed to load pretrained weights: {str(e)}")
        else:
            logging.info(
                f"Skipping pretrained weights: input has {input_channels} channels (not RGB)")


def check_num_classes(func):
    """Decorator to check if num_classes is provided"""
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)
    return wrapper


@check_num_classes
def get_mobilenet(version: str, pretrained: bool = False, input_channels: int = 3,
                  num_classes: int = None, robust_method: Optional[BaseRobustMethod] = None):
    """
    Factory method to obtain a MobileNet model.

    Args:
        version: 'v2' or 'v3small' - MobileNet version to use
        pretrained: Whether to load pretrained weights
        input_channels: Number of input image channels
        num_classes: Number of classes for classification
        robust_method: Optional robust method module

    Returns:
        MobileNet model with the specified configuration
    """
    version = version.lower()
    if version == 'v2':
        model = MobileNetV2(
            num_classes=num_classes,
            input_channels=input_channels,
            robust_method=robust_method
        )
        logging.info(f"Created MobileNetV2 model with {num_classes} classes")
    elif version == 'v3small':
        model = MobileNetV3Small(
            num_classes=num_classes,
            input_channels=input_channels,
            robust_method=robust_method
        )
        logging.info(
            f"Created MobileNetV3-Small model with {num_classes} classes")
    else:
        raise ValueError(
            f"Unsupported MobileNet version: {version}. Use 'v2' or 'v3small'.")

    if pretrained:
        model.load_pretrained_weights(input_channels)

    return model

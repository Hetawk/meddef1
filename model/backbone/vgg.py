import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import List, Dict, Optional, Tuple, Union
from model.attention.base_robust_method import BaseRobustMethod

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c7685960.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int, input_channels: int = 3, pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None, depth: int = 16):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if pretrained:
            self.load_pretrained_weights(depth)

        self.robust_method = robust_method  # Initialize the robust method module

        # Adjust the first layer for custom input channels
        if input_channels != 3:
            self.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if self.robust_method:
            x, _ = self.robust_method(x, x, x)  # Apply robust method
            return x  # Maintain 2D shape
        x = self.classifier(x)
        return x

    def load_pretrained_weights(self, depth: int):
        url = model_urls.get(f'vgg{depth}')
        if url is None:
            raise ValueError(f"No pretrained model available for VGG{depth}")

        pretrained_dict = load_state_dict_from_url(url, progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

def make_vgg(cfg: List[Union[int, str]], batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3  # Default input channels

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
            in_channels = v

    return nn.Sequential(*layers)

def get_vgg(depth: int, pretrained: bool = False, input_channels: int = 3, num_classes: int = 1000,
            robust_method: Optional[BaseRobustMethod] = None) -> VGG:
    depth_to_cfg = {
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11
        13: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13
        16: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
        19: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG19
    }
    if depth not in depth_to_cfg:
        raise ValueError(f"Unsupported VGG depth: {depth}")

    cfg = depth_to_cfg[depth]
    features = make_vgg(cfg)
    return VGG(features, num_classes=num_classes, pretrained=pretrained, input_channels=input_channels,
               robust_method=robust_method, depth=depth)
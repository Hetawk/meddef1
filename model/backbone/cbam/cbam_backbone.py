#!/usr/bin/env python3
# cbam_backbone.py
from torch import nn
from torch.nn import functional as F
from .cbam import CBAM

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.norm = norm
        if norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)
        else:
            self.norm_layer = None

    def forward(self, x):
        x = super().forward(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        return x

class BasicStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class CBAMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, norm=True):
        super().__init__()
        self.shortcut = None if in_channels == out_channels else Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=norm)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, norm=norm)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm=norm)
        self.cbam = CBAM(out_channels, reduction_ratio=16)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out = self.cbam(out)
        out += shortcut
        out = F.relu(out)
        return out

class CBAMBottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, bottleneck_channels=None, stride=1, num_groups=1, norm=True, stride_in_1x1=False, dilation=1):
        super().__init__()
        self.shortcut = None if in_channels == out_channels else Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=norm)
        bottleneck_channels = bottleneck_channels or out_channels // 4
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=norm)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=norm)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=norm)
        self.cbam = CBAM(out_channels, reduction_ratio=16)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out = self.cbam(out)
        out += shortcut
        out = F.relu(out)
        return out

class CBAMResNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = BasicStem(in_channels, 64)
        self.res2 = self.make_layer(CBAMBasicBlock, 64, 64, num_blocks=2, stride=1, norm=True)
        self.res3 = self.make_layer(CBAMBottleneckBlock, 64 * CBAMBasicBlock.expansion, 128, num_blocks=3, stride=2, norm=True, bottleneck_channels=128)
        self.res4 = self.make_layer(CBAMBottleneckBlock, 128 * CBAMBottleneckBlock.expansion, 256, num_blocks=4, stride=2, norm=True, bottleneck_channels=256)
        self.res5 = self.make_layer(CBAMBottleneckBlock, 256 * CBAMBottleneckBlock.expansion, 512, num_blocks=6, stride=2, norm=True, bottleneck_channels=512)

    def make_layer(self, block, in_channels, out_channels, num_blocks, stride=1, norm=True, bottleneck_channels=None):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(in_channels, out_channels, stride=stride, norm=norm, bottleneck_channels=bottleneck_channels))
            else:
                layers.append(block(out_channels * block.expansion, out_channels, stride=1, norm=norm))
            in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        return x
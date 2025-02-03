#!/usr/bin/env python3
# cbamsa.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.backbone.cbam.cbam import CBAM
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, ResNet
from detectron2.modeling.backbone.resnet import BasicStem, weight_init
from torch.utils.checkpoint import checkpoint

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def self_attention(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

    def forward(self, x):
        x.requires_grad_(True)  # Ensure requires_grad is True
        return checkpoint(self.self_attention, x)

class CBAMBasicBlock(CNNBlockBase):
    expansion = 1

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        super().__init__(in_channels, out_channels, stride)
        self.shortcut = None if in_channels == out_channels else Conv2d(in_channels, out_channels, kernel_size=1,
                                                                        stride=stride, bias=False,
                                                                        norm=get_norm(norm, out_channels))
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                            norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                            norm=get_norm(norm, out_channels))
        self.cbam = CBAM(out_channels, reduction_ratio=16)
        self.self_attention = SelfAttention(out_channels)
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out = self.cbam(out)
        out = self.self_attention(out)
        out += shortcut
        out = F.relu_(out)
        return out

class CBAMBottleneckBlock(CNNBlockBase):
    expansion = 4

    def __init__(self, in_channels, out_channels, *, bottleneck_channels=None, stride=1, num_groups=1, norm="BN", stride_in_1x1=False, dilation=1):
        super().__init__(in_channels, out_channels, stride)
        self.shortcut = None if in_channels == out_channels else Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        bottleneck_channels = bottleneck_channels or out_channels // 4
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels))
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))
        self.cbam = CBAM(out_channels, reduction_ratio=16)
        self.self_attention = SelfAttention(out_channels)
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        out = self.conv3(out)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out = self.cbam(out)
        out = self.self_attention(out)
        out += shortcut
        out = F.relu_(out)
        return out

# Check if CBAMResNet is already registered
if "CBAMResNet" not in BACKBONE_REGISTRY:
    @BACKBONE_REGISTRY.register()
    class CBAMResNet(ResNet):
        def __init__(self, cfg, input_shape):
            super().__init__(cfg, input_shape)
            self.stem = BasicStem(cfg, input_shape)
            self.res2 = self.make_layer(CBAMBasicBlock, self.stem.out_channels, 64, num_blocks=2, stride=1, norm="BN")
            self.res3 = self.make_layer(CBAMBottleneckBlock, 64 * CBAMBasicBlock.expansion, 128, num_blocks=3, stride=2, norm="BN", bottleneck_channels=128)
            self.res4 = self.make_layer(CBAMBottleneckBlock, 128 * CBAMBottleneckBlock.expansion, 256, num_blocks=4, stride=2, norm="BN", bottleneck_channels=256)
            self.res5 = self.make_layer(CBAMBottleneckBlock, 256 * CBAMBottleneckBlock.expansion, 512, num_blocks=6, stride=2, norm="BN", bottleneck_channels=512)

        def make_layer(self, block, in_channels, out_channels, num_blocks, stride=1, norm="BN", bottleneck_channels=None):
            layers = []
            for i in range(num_blocks):
                if i == 0:
                    if block == CBAMBottleneckBlock:
                        layers.append(block(in_channels, out_channels, stride=stride, norm=norm, bottleneck_channels=bottleneck_channels))
                    else:
                        layers.append(block(in_channels, out_channels, stride=stride, norm=norm))
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
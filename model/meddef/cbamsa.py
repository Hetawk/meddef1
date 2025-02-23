import torch
import torch.nn as nn
from torch.nn import functional as F
from model.backbone.cbam.cbam import CBAM
from model.attention.base.self_attention import SelfAttention
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, ResNet
from detectron2.modeling.backbone.resnet import BasicStem, weight_init


class CBAMBasicBlock(CNNBlockBase):
    expansion = 1

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        super().__init__(in_channels, out_channels, stride)
        # Basic block components
        self.shortcut = None if in_channels == out_channels else Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride,
            bias=False, norm=get_norm(norm, out_channels)
        )
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False, norm=get_norm(norm, out_channels)
        )
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False, norm=get_norm(norm, out_channels)
        )

        # Attention modules
        self.cbam = CBAM(out_channels, reduction_ratio=16)
        # Initialize attention with proper dimensions
        self.self_attention = SelfAttention(
            in_dim=out_channels,
            key_dim=out_channels // 8,
            query_dim=out_channels // 8,
            value_dim=out_channels
        )

        # Initialize weights
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x

        # Convolution path
        out = F.relu_(self.conv1(x))
        out = self.conv2(out)

        # Apply CBAM
        out = self.cbam(out)

        # Process for self attention
        B, C, H, W = out.shape
        # Reshape to (B, L, D) where L is sequence length (H*W) and D is dimension (C)
        out_flat = out.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Pass the same tensor as query, key, and value
        attended = self.self_attention(
            query=out_flat, key=out_flat, value=out_flat)

        # Reshape back to (B, C, H, W)
        out = attended.permute(0, 2, 1).view(B, C, H, W)

        # Residual connection and activation
        out += identity
        out = F.relu_(out)
        return out


class CBAMBottleneckBlock(CNNBlockBase):
    expansion = 4

    def __init__(self, in_channels, out_channels, *, bottleneck_channels=None,
                 stride=1, num_groups=1, norm="BN", stride_in_1x1=False, dilation=1):
        super().__init__(in_channels, out_channels, stride)

        # Compute channels
        bottleneck_channels = bottleneck_channels or out_channels // 4
        out_channels_expanded = out_channels * self.expansion

        # Shortcut connection
        self.shortcut = None if in_channels == out_channels_expanded else Conv2d(
            in_channels, out_channels_expanded, kernel_size=1,
            stride=stride, bias=False, norm=get_norm(norm, out_channels_expanded)
        )

        # Main layers
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(
            in_channels, bottleneck_channels, kernel_size=1,
            stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels)
        )
        self.conv2 = Conv2d(
            bottleneck_channels, bottleneck_channels, kernel_size=3,
            stride=stride_3x3, padding=1 * dilation, bias=False,
            groups=num_groups, dilation=dilation,
            norm=get_norm(norm, bottleneck_channels)
        )
        self.conv3 = Conv2d(
            bottleneck_channels, out_channels_expanded, kernel_size=1,
            bias=False, norm=get_norm(norm, out_channels_expanded)
        )

        # Attention modules
        self.cbam = CBAM(out_channels_expanded, reduction_ratio=16)
        # Initialize attention with proper dimensions
        self.self_attention = SelfAttention(
            in_dim=out_channels_expanded,
            key_dim=out_channels_expanded // 8,
            query_dim=out_channels_expanded // 8,
            value_dim=out_channels_expanded
        )

        # Initialize weights
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x

        # Convolution path
        out = F.relu_(self.conv1(x))
        out = F.relu_(self.conv2(out))
        out = self.conv3(out)

        # Apply CBAM
        out = self.cbam(out)

        # Process for self attention
        B, C, H, W = out.shape
        out_flat = out.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Pass the same tensor as query, key, and value
        attended = self.self_attention(
            query=out_flat, key=out_flat, value=out_flat)

        # Reshape back to (B, C, H, W)
        out = attended.permute(0, 2, 1).view(B, C, H, W)

        # Residual connection and activation
        out += identity
        out = F.relu_(out)
        return out


# Register CBAMResNet if not already registered
if "CBAMResNet" not in BACKBONE_REGISTRY:
    @BACKBONE_REGISTRY.register()
    class CBAMResNet(ResNet):
        def __init__(self, cfg, input_shape):
            super().__init__(cfg, input_shape)
            self.stem = BasicStem(cfg, input_shape)
            self.res2 = self.make_layer(
                CBAMBasicBlock, self.stem.out_channels, 64,
                num_blocks=2, stride=1, norm="BN"
            )
            self.res3 = self.make_layer(
                CBAMBottleneckBlock, 64 * CBAMBasicBlock.expansion, 128,
                num_blocks=3, stride=2, norm="BN", bottleneck_channels=128
            )
            self.res4 = self.make_layer(
                CBAMBottleneckBlock, 128 * CBAMBottleneckBlock.expansion, 256,
                num_blocks=4, stride=2, norm="BN", bottleneck_channels=256
            )
            self.res5 = self.make_layer(
                CBAMBottleneckBlock, 256 * CBAMBottleneckBlock.expansion, 512,
                num_blocks=6, stride=2, norm="BN", bottleneck_channels=512
            )

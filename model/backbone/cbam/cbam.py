#!/usr/bin/env python3
# cbam.py

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import weight_init
from .utils import Flatten, logsumexp_2d

__all__ = ["CBAM"]

class CBAM(nn.Module):
    """CBAM Layer with both channel and spatial attention."""
    def __init__(self, gate_channels: int, reduction_ratio: int = 16, pool_types: List[str] = ["avg", "max"], use_spatial: bool = True) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.spatial_gate = SpatialGate() if use_spatial else None

    def forward(self, x):
        x_out = self.channel_gate(x)
        if self.spatial_gate:
            x_out = self.spatial_gate(x_out)
        return x_out


class BasicConv(nn.Module):
    """Basic Convolutional Block with optional BatchNorm and ReLU."""
    def __init__(self, in_planes: int, out_planes: int, kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, dilation: Union[int, Tuple[int]] = 1, groups: int = 1, relu: bool = True, bn: bool = True, bias: bool = False, **kwargs):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        weight_init.c2_msra_fill(self.conv)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """Channel Attention Gate."""
    def __init__(self, gate_channels: int, reduction_ratio: int = 16, pool_types: List[str] = ["avg", "max"], **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_init.c2_msra_fill(m)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    """Channel pooling (max and mean)."""
    def forward(self, x):
        assert x.dim() >= 3
        return torch.cat((torch.max(x, -3)[0].unsqueeze(-3), torch.mean(x, -3).unsqueeze(-3)), dim=-3)


class SpatialGate(nn.Module):
    """Spatial Attention Gate."""
    def __init__(self, kernel_size: Union[int, Tuple[int]] = 7) -> None:
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

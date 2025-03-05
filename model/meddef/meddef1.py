import torch
import torch.nn as nn
from model.attention.base.self_attention import SelfAttention
from model.attention.base_robust_method import BaseRobustMethod
from model.defense.multi_scale import DefenseModule
from model.backbone.resnet import BasicBlock, Bottleneck


class ResNetSelfAttention(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3, robust_method: BaseRobustMethod = None):
        super(ResNetSelfAttention, self).__init__()
        self.in_channels = 64
        self.block_expansion = block.expansion
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Add defense module after layer4
        self.defense = DefenseModule(512 * self.block_expansion)

        channels = 512 * self.block_expansion
        # Fix self-attention dimensions
        self.self_attention = SelfAttention(
            in_dim=channels,  # Input dimension
            key_dim=channels,  # Keep full dimension for key
            query_dim=channels,  # Keep full dimension for query
            value_dim=channels  # Keep full dimension for value
        )

        # Adjust fusion layer for the new dimensions
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)
        self.robust_method = robust_method

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply defense module
        defended = self.defense(x)

        # Process features properly
        B, C, H, W = defended.shape

        # Global pooling path
        spatial_features = self.avgpool(defended)
        spatial_features = torch.flatten(spatial_features, 1)  # [B, C]

        # Self-attention path with proper reshaping
        flat_features = defended.flatten(2)  # [B, C, HW]
        flat_features = flat_features.permute(0, 2, 1)  # [B, HW, C]

        # Apply self-attention and handle tuple return
        attended_features, _ = self.self_attention(
            query=flat_features,
            key=flat_features,
            value=flat_features
        )  # Now properly unpacking tuple

        # Average over spatial dimensions
        attended_features = attended_features.mean(1)  # [B, C]

        # Combine features
        combined = torch.cat([spatial_features, attended_features], dim=1)
        fused = self.fusion(combined)

        # Apply robust method if available
        if self.robust_method is not None:
            fused, _ = self.robust_method(
                fused.unsqueeze(2),
                fused.unsqueeze(2),
                fused.unsqueeze(2)
            )
            fused = fused.squeeze(2)

        # Final classification
        out = self.fc(fused)
        return out


def get_meddef1(depth: float, input_channels=3, num_classes=None, robust_method: BaseRobustMethod = None):
    depth_to_block_layers = {
        1.0: (BasicBlock, (2, 2, 2, 2)),
        1.1: (BasicBlock, (3, 4, 6, 3)),
        1.2: (Bottleneck, (3, 4, 6, 3)),
        1.3: (Bottleneck, (3, 4, 23, 3)),
        1.4: (Bottleneck, (3, 8, 36, 3))
    }
    if depth not in depth_to_block_layers:
        raise ValueError(f"Unsupported meddef1 depth: {depth}")
    block, layers = depth_to_block_layers[depth]
    return ResNetSelfAttention(block, layers, num_classes=num_classes, input_channels=input_channels, robust_method=robust_method)

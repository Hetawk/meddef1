import torch
import torch.nn as nn
from model.attention.base.self_attention import SelfAttention  # your flexible self-attention
from model.attention.base_robust_method import BaseRobustMethod
from model.backbone.resnet import BasicBlock, Bottleneck  # standard ResNet blocks

class ResNetSelfAttention(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3, robust_method: BaseRobustMethod = None):
        super(ResNetSelfAttention, self).__init__()
        self.in_channels = 64
        self.block_expansion = block.expansion
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)

        # Updated self_attention instantiation with proper dimensions.
        self.self_attention = SelfAttention(512 * self.block_expansion,
                                            512 * self.block_expansion,
                                            512 * self.block_expansion)
        self.robust_method = robust_method

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Standard ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)       # shape: [batch, channels, 1, 1]
        x = torch.flatten(x, 1)   # shape: [batch, channels]

        # Apply self-attention (unsqueeze to add sequence dim, then squeeze back)
        x_seq = x.unsqueeze(1)
        x_att, _ = self.self_attention(x_seq, x_seq, x_seq)
        x = x_att.squeeze(1)

        if self.robust_method:
            x, _ = self.robust_method(x, x, x)

        x = self.fc(x)
        return x

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

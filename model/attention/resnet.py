import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Tuple, Dict, Union, Optional
from model.attention.base_robust_method import BaseRobustMethod

# Define URLs for loading pretrained weights for various ResNet depths
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34.
    Supports optional application of robust attention mechanisms.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 robust_method: Optional[BaseRobustMethod] = None):
        super(BasicBlock, self).__init__()
        # First convolutional layer in the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer in the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer, only used if we need to match the output dimensions to the residual
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Optional attention mechanism, applied after the first convolution
        self.robust_method = robust_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Save the input tensor for the residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply attention mechanism if provided
        if self.robust_method:
            out, _ = self.robust_method(out, out, out)  # Self-attention within residual block

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if necessary to match dimensions for residual addition
        if self.downsample is not None:
            residual = self.downsample(x)

        # Residual connection: add the input to the output of the block
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50, ResNet-101, and ResNet-152.
    Supports optional robust attention mechanisms.
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 robust_method: Optional[BaseRobustMethod] = None):
        super(Bottleneck, self).__init__()

        # Three convolutional layers in the bottleneck block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Downsample layer, used if we need to match output dimensions to the residual
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # Optional attention mechanism, applied after the second convolution layer
        self.robust_method = robust_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Save the input tensor for the residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply attention mechanism if provided
        if self.robust_method:
            out, _ = self.robust_method(out, out, out)  # Self-attention in bottleneck block

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply downsampling if necessary to match dimensions for residual addition
        if self.downsample is not None:
            residual = self.downsample(x)

        # Residual connection: add the input to the output of the block
        out += residual
        out = self.relu(out)
        return out


class ResNetModel(nn.Module):
    """
    ResNet model supporting flexible attention mechanisms.
    """

    def __init__(self, block: Union[Type[BasicBlock], Type[Bottleneck]], layers: Tuple[int, int, int, int],
                 num_classes: int, input_channels: int = 3, pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None, global_attention: Optional[BaseRobustMethod] = None):
        super(ResNetModel, self).__init__()

        # Initial convolutional layer, batch normalization, and pooling
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers, with optional robust attention in deeper layers
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, robust_method=robust_method)  # Attention here
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, robust_method=robust_method)  # Attention here

        # Adaptive pooling and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Load pretrained weights if specified
        if pretrained:
            self.load_pretrained_weights(block, layers, num_classes, input_channels)

        # Optional global attention layer applied after all feature extraction layers
        self.global_attention = global_attention

    def make_layer(self, block, out_channels, blocks, stride=1, robust_method=None):
        """
        Create a residual layer with multiple blocks. Optionally apply attention within each block.
        """
        layers = []

        # First block in the layer, with specified stride and optional robust attention
        layers.append(block(self.in_channels, out_channels, stride, robust_method))
        self.in_channels = out_channels * block.expansion

        # Additional blocks in the layer
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, robust_method=robust_method))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet model.
        Optionally applies global attention after the convolutional layers.
        """
        x = self.forward_without_fc(x)

        # Apply global attention if provided
        if self.global_attention:
            x, _ = self.global_attention(x, x, x)

        x = x.view(x.size(0), -1)  # Flatten to 2D [batch_size, channels]
        x = self.fc(x)
        return x

    def forward_without_fc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model without the fully connected layer.
        This is useful if you want the feature representation before classification.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # Attention can be applied here if `robust_method` is provided
        x = self.layer4(x)  # Attention can be applied here if `robust_method` is provided
        x = self.avgpool(x)
        return x

    def load_pretrained_weights(self, block, layers, num_classes, input_channels):
        """
        Load pretrained weights for the specified architecture if available.
        Also adjusts the first convolution layer and the fully connected layer as needed.
        """
        # Map architecture to its corresponding pretrained model URL
        depth_to_url = {
            (BasicBlock, (2, 2, 2, 2)): model_urls['resnet18'],
            (BasicBlock, (3, 4, 6, 3)): model_urls['resnet34'],
            (Bottleneck, (3, 4, 6, 3)): model_urls['resnet50'],
            (Bottleneck, (3, 4, 23, 3)): model_urls['resnet101'],
            (Bottleneck, (3, 8, 36, 3)): model_urls['resnet152'],
        }
        url = depth_to_url.get((block, layers))
        if url is None:
            raise ValueError("No pretrained model available for the specified architecture.")

        # Load and filter pretrained state dictionary to exclude the final fully connected layer
        pretrained_dict = load_state_dict_from_url(url, progress=True)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # Modify the initial convolution and final fully connected layer if input channels or class count differ
        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fc = nn.Linear(512 * block.expansion, num_classes)


# Usage example for constructing a ResNet model
def get_resnet(depth: int, pretrained: bool = False, input_channels: int = 3, num_classes: int = None,
               robust_method: Optional[BaseRobustMethod] = None,
               global_attention: Optional[BaseRobustMethod] = None) -> ResNetModel:
    """
    Factory function to create a ResNet model with optional robust attention.
    """
    depth_to_block_layers = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }
    if depth not in depth_to_block_layers:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    block, layers = depth_to_block_layers[depth]
    return ResNetModel(block, layers, num_classes=num_classes, pretrained=pretrained, input_channels=input_channels,
                       robust_method=robust_method, global_attention=global_attention)


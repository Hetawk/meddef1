import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Type
from model.attention.base_robust_method import BaseRobustMethod

try:
    import ultralytics
    from ultralytics.nn.modules import C2f, Conv, SPPF, Bottleneck
    from ultralytics.nn.tasks import DetectionModel
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics package not found. YOLOv8 models will not be available. "
                    "Install using: pip install ultralytics")


class YOLOv8Backbone(nn.Module):
    """
    YOLOv8 model adapted to be used as a backbone for classification tasks.

    This implementation extracts the feature extractor part of YOLOv8 and adds
    a classification head on top.
    """

    def __init__(self,
                 variant: str = 'small',
                 num_classes: int = 1000,
                 input_channels: int = 3,
                 pretrained: bool = False,
                 robust_method: Optional[BaseRobustMethod] = None):
        """
        Args:
            variant: YOLOv8 variant ('nano', 'small', 'medium', 'large', 'xlarge')
            num_classes: Number of output classes
            input_channels: Number of input channels
            pretrained: Whether to load pretrained weights
            robust_method: Optional robust method to apply
        """
        super(YOLOv8Backbone, self).__init__()

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics package is required for YOLOv8. Install using: pip install ultralytics")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.robust_method = robust_method
        self.variant = variant

        # Get YOLO base model
        self.model = self._build_model(variant, pretrained)

        # Determine final feature dimension by running a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            features = self._forward_backbone(dummy_input)
            feature_dim = features.shape[1]

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        # Initialize classification head
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # Handle non-RGB inputs if pretrained
        if pretrained and input_channels != 3:
            self._adapt_input_channels()

        logging.info(
            f"Created YOLOv8-{variant} backbone with {num_classes} classes")

    def _build_model(self, variant: str, pretrained: bool) -> nn.Module:
        """Build YOLOv8 model based on variant"""
        variant_map = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt',
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt',
            'xlarge': 'yolov8x.pt'
        }

        if variant not in variant_map:
            raise ValueError(
                f"Invalid YOLOv8 variant: {variant}. Choose from: {list(variant_map.keys())}")

        model_path = variant_map[variant]

        if pretrained:
            # Load pretrained YOLOv8 model
            model = ultralytics.YOLO(model_path)
            # Access the PyTorch model
            backbone = model.model
        else:
            # Initialize from scratch using Ultralytics code
            backbone = DetectionModel(model_path.replace('.pt', '.yaml'))

        return backbone

    def _adapt_input_channels(self):
        """
        Adapt the first convolution layer to handle non-RGB inputs
        while preserving pretrained weights when possible
        """
        # Get the first layer (should be a Conv)
        first_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, Conv) and module.conv.in_channels == 3:
                first_layer = module
                break

        if not first_layer:
            logging.warning("Could not find first convolutional layer")
            return

        # Get the Conv2d layer
        original_conv = first_layer.conv

        # Create a new Conv2d layer with desired input channels
        new_conv = nn.Conv2d(
            self.input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )

        # Initialize the new layer
        nn.init.kaiming_normal_(
            new_conv.weight, mode='fan_out', nonlinearity='relu')

        # If possible, reuse weights for the existing channels
        if self.input_channels > 3:
            # For more channels, copy the original weights and initialize the rest
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
        else:
            # For fewer channels, use a subset of the original weights
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    original_conv.weight[:, :self.input_channels, :, :])

        # Replace the conv layer
        first_layer.conv = new_conv

        logging.info(
            f"Adapted input layer to handle {self.input_channels} channels")

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the YOLOv8 backbone"""
        # YOLOv8 forward goes through the backbone model
        # We want to stop at the last feature map before detection heads
        for i, m in enumerate(self.model.model):
            if i == 9:  # This is the index before detection heads in YOLOv8
                return x
            x = m(x)
        return x

    def forward_without_fc(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head"""
        features = self._forward_backbone(x)
        pooled = self.avgpool(features)  # Shape: [B, C, 1, 1]
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional robust method application"""
        x = self.forward_without_fc(x)

        if self.robust_method:
            # Apply robust method
            x, _ = self.robust_method(x, x, x)
            return x  # Return 4D tensor for compatibility with attention modules
        else:
            # Standard forward path with classification head
            return self.classifier(x)


class YOLOv8FeatureExtractor(nn.Module):
    """
    Feature extractor for YOLOv8 that returns intermediate feature maps
    for multi-scale feature fusion or other downstream tasks
    """

    def __init__(self, variant='small', pretrained=False, input_channels=3):
        super(YOLOv8FeatureExtractor, self).__init__()

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics package is required for YOLOv8")

        self.model = self._build_model(variant, pretrained)

        # Handle non-RGB inputs
        if input_channels != 3:
            self._adapt_input_channels(input_channels)

    def _build_model(self, variant, pretrained):
        variant_map = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt',
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt',
            'xlarge': 'yolov8x.pt'
        }

        if variant not in variant_map:
            raise ValueError(f"Invalid YOLOv8 variant: {variant}")

        model_path = variant_map[variant]

        if pretrained:
            model = ultralytics.YOLO(model_path)
            backbone = model.model
        else:
            backbone = DetectionModel(model_path.replace('.pt', '.yaml'))

        return backbone

    def _adapt_input_channels(self, input_channels):
        # Similar to the implementation in YOLOv8Backbone
        for name, module in self.model.named_modules():
            if isinstance(module, Conv) and module.conv.in_channels == 3:
                original_conv = module.conv
                new_conv = nn.Conv2d(input_channels, original_conv.out_channels,
                                     kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=False if original_conv.bias is None else True)

                if input_channels > 3:
                    with torch.no_grad():
                        new_conv.weight[:, :3, :, :] = original_conv.weight
                else:
                    with torch.no_grad():
                        new_conv.weight = nn.Parameter(
                            original_conv.weight[:, :input_channels, :, :])

                module.conv = new_conv
                break

    def forward(self, x):
        """
        Returns feature maps at different scales
        from the YOLOv8 backbone
        """
        features = []
        # Extract features at key points in the model
        for i, m in enumerate(self.model.model):
            x = m(x)
            # Collect features at key points (adjust indices based on the specific architecture)
            if i in [4, 6, 9]:  # P3, P4, P5 feature levels
                features.append(x)

        return features


def check_num_classes(func):
    """Decorator to check if num_classes is provided"""
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)
    return wrapper


@check_num_classes
def get_yolov8(variant: str = 'small',
             pretrained: bool = False,
             input_channels: int = 3,
             num_classes: int = None,
             robust_method: Optional[BaseRobustMethod] = None) -> YOLOv8Backbone:
    """
    Get a YOLOv8 backbone model configured for classification

    Args:
        variant: YOLOv8 variant - 'nano', 'small', 'medium', 'large', or 'xlarge'
        pretrained: Whether to load pretrained weights
        input_channels: Number of input channels
        num_classes: Number of output classes
        robust_method: Optional robust method to apply

    Returns:
        YOLOv8 model configured as a backbone for classification
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Ultralytics package is required. Install with: pip install ultralytics")

    valid_variants = ['nano', 'small', 'medium', 'large', 'xlarge']
    if variant.lower() not in valid_variants:
        raise ValueError(
            f"Invalid YOLOv8 variant: {variant}. Choose from: {valid_variants}")

    model = YOLOv8Backbone(
        variant=variant.lower(),
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained,
        robust_method=robust_method
    )

    logging.info(f"Created YOLOv8-{variant} model with {num_classes} classes")

    return model


# Convenience functions for different YOLOv8 variants
def yolo_nano(pretrained=False, **kwargs):
    return get_yolov8('nano', pretrained=pretrained, **kwargs)


def yolo_small(pretrained=False, **kwargs):
    return get_yolov8('small', pretrained=pretrained, **kwargs)


def yolo_medium(pretrained=False, **kwargs):
    return get_yolov8('medium', pretrained=pretrained, **kwargs)


def yolo_large(pretrained=False, **kwargs):
    return get_yolov8('large', pretrained=pretrained, **kwargs)


def yolo_xlarge(pretrained=False, **kwargs):
    return get_yolov8('xlarge', pretrained=pretrained, **kwargs)

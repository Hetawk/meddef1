import torch
import torch.nn as nn
from model.backbone.resnet import ResNetModel, BasicBlock, Bottleneck
from model.base_robust_method import BaseRobustMethod

class MSARNet(nn.Module):
    def __init__(self, depth: int, num_classes: int, input_channels: int = 3, pretrained: bool = False,
                 robust_method_type: str = None, robust_method_params: dict = None):
        super(MSARNet, self).__init__()
        self.robust_method_type = robust_method_type
        self.robust_method_params = robust_method_params or {}

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

        robust_method = None
        if robust_method_type:
            robust_method = BaseRobustMethod(robust_method_type, 512 * block.expansion, num_classes, **robust_method_params)

        self.resnet = ResNetModel(
            block=block,
            layers=layers,
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            robust_method=robust_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
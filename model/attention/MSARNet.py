import torch
import torch.nn as nn
from model.backbone.resnet import ResNetModel, BasicBlock, Bottleneck
from model.attention.base_robust_method import BaseRobustMethod
from model.attention.attention import Attention  # Import the flexible Attention class

class MSARNet(nn.Module):
    def __init__(self, depth: int, num_classes: int, input_channels: int = 3,
                 pretrained: bool = False, robust_method_params: dict = None):
        """
        Multi-Scale Attention ResNet (MSARNet) model
        """
        super(MSARNet, self).__init__()

        # Set default robust method and attention types
        robust_method_params = robust_method_params or {}
        self.robust_method_type = robust_method_params.get('method_type', 'attention')
        self.attention_types = robust_method_params.get('attention_types', ['spatial', 'self', 'global'])

        # Define ResNet depth and block configurations
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
        value_dim = 512 * block.expansion  # Define value dimension based on expansion

        # Initialize BaseRobustMethod
        self.robust_method = BaseRobustMethod(
            method_type=self.robust_method_type,
            input_dim=value_dim,
            output_dim=num_classes,
            value_dim=value_dim,
            attention_types=self.attention_types,
            **robust_method_params
        )

        # Initialize ResNetModel without robust method
        self.resnet = ResNetModel(
            block=block,
            layers=layers,
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            robust_method=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"MSARNet: - Input shape to ResNet: {x.shape}")
        x = self.resnet.forward_without_fc(x)  # Use the method without fc
        print(f"MSARNet: - Output shape from ResNet: {x.shape}")

        # Handle robust method processing
        if self.robust_method:
            if len(x.shape) != 4:
                print(f"MSARNet: - Unexpected output shape from ResNet for robust method, expected 4D, got {x.shape}")
                raise ValueError("MSARNet: - Expected 4D tensor from ResNet when robust method is used")

            batch_size, channels, height, width = x.shape
            print(f"MSARNet: - Shape before applying attention: {x.shape}")

            attention_outputs = []
            attention_weights_list = []

            # Reshape for self attention
            if 'self' in self.attention_types:
                if height == 1 and width == 1:
                    print(f"MSARNet: - Reshaping tensor for self attention.")
                    x_reshaped = x.view(batch_size, channels)  # Shape: [batch_size, channels]
                    x_reshaped = x_reshaped.unsqueeze(1)  # Shape: [batch_size, 1, channels]
                else:
                    x_reshaped = x.view(batch_size, channels, -1)  # Shape: [batch_size, channels, height * width]
                    x_reshaped = x_reshaped.permute(0, 2, 1)  # Shape: [batch_size, height * width, channels]

                self_output, self_weights = self.robust_method(x_reshaped, x_reshaped, x_reshaped)
                # Reshape back to original dimensions if necessary
                if height > 1 and width > 1:
                    self_output = self_output.permute(0, 2, 1).view(batch_size, channels, height,
                                                                    width)  # Shape: [batch_size, channels, height, width]
                attention_outputs.append(self_output)
                attention_weights_list.append(self_weights)

            # Spatial Attention (if needed)
            if 'spatial' in self.attention_types:
                print(f"MSARNet: - Applying spatial attention.")
                spatial_output, spatial_weights = self.robust_method(x, x, x)  # Apply robust method
                attention_outputs.append(spatial_output)
                attention_weights_list.append(spatial_weights)

            # Global Attention
            if 'global' in self.attention_types:
                print(f"MSARNet: - Applying global attention.")
                global_output, global_weights = self.robust_method(x, x, x)  # Assuming global can also use the same x
                attention_outputs.append(global_output)
                attention_weights_list.append(global_weights)

            # Combine the outputs
            if attention_outputs:
                x = torch.cat(attention_outputs, dim=1)  # Concatenate along the channel dimension
                print(f"MSARNet: - Shape after combining attention outputs: {x.shape}")
                x = x.view(batch_size, -1)  # Flatten for final layer
                print(f"MSARNet: - Shape after flattening: {x.shape}")

        return x





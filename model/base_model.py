import logging
import os

import torch
from model.alexnet_model import AlexNetModel
from model.resnet_model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.densenet_model import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from model.resnext_model import ResNeXt50, ResNeXt101, ResNeXt152
from model.mobilenet_model import MobileNetV2Model, MobileNetV3SmallModel
from model.hybrid_models import HybridResNetDenseNet
from model.transformer_model import TransformerModel
from model.conditional_diffusion_model import ConditionalDiffusionModel

class ModelLoader:
    def __init__(self, device):
        self.device = device
        self.models_dict = {
            # 'alexnet': AlexNetModel,
            # 'resnet18': ResNet18,
            # 'resnet34': ResNet34,
            'resnet50': ResNet50,
            # 'resnet101': ResNet101,
            # 'resnet152': ResNet152,
            # 'transformer': TransformerModel,
            # 'conditional_diffusion': ConditionalDiffusionModel,
            'densenet121': DenseNet121,
            # 'densenet169': DenseNet169,
            # 'densenet201': DenseNet201,
            # 'densenet264': DenseNet264,
            # 'resnext50': ResNeXt50,
            # 'resnext101': ResNeXt101,
            # 'resnext152': ResNeXt152,
            # 'mobilenet_v2': MobileNetV2Model,
            # 'mobilenet_v3_small': MobileNetV3SmallModel,
            # 'hybrid_resnet_densenet': HybridResNetDenseNet,
        }
        logging.info("ModelLoader initialized with models: " + ", ".join(self.models_dict.keys()))

    def get_model(self, model_name, input_channels=3, pretrained=False):
        logging.info(f"Getting model {model_name} with {input_channels} input channels.")
        if model_name in self.models_dict:
            model_class = self.models_dict[model_name]
            if model_name.startswith('resnet') or model_name.startswith('densenet') or model_name.startswith('resnext'):
                model = model_class(pretrained=pretrained, input_channels=input_channels)
            elif model_name.startswith('mobilenet'):
                model = model_class(pretrained=pretrained, input_channels=input_channels)
            elif model_name == 'hybrid_resnet_densenet':
                model = model_class(num_blocks_resnet=[3, 4, 6, 3], num_blocks_densenet=[6, 12, 32, 32], pretrained_resnet=pretrained, pretrained_densenet=pretrained, input_channels=input_channels)
            else:
                model = model_class(input_channels=input_channels)

            if torch.cuda.is_available():
                model = model.to(self.device)
            return model
        else:
            raise ValueError(f"Model {model_name} not recognized.")

    def load_pretrained_model(self, model_name, load_task, dataset_name):
        model_class = self.models_dict.get(model_name)
        if model_class is None:
            raise ValueError(f"Model {model_name} not found.")
        model = model_class()  # Initialize the model
        model = model.to(self.device)  # Move the model to the appropriate device

        # Load the state dict from a file
        model_path = f"out/{load_task}/{dataset_name}/save_model/best_{model_name}_{load_task}.pth"
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError(f"No saved model found at {model_path}")

        return model

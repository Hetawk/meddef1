import logging
import os

import torch
from model.alexnet_model import AlexNetModel
from model.meddef import MedDef
from model.resnet_model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.densenet_model import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from model.resnext_model import ResNeXt50, ResNeXt101_32x8d, ResNeXt101_64x4d
from model.mobilenet_model import MobileNetV2Model, MobileNetV3SmallModel
from model.hybrid_models import HybridResNetDenseNet
from model.transformer_model import TransformerModel
from model.conditional_diffusion_model import ConditionalDiffusionModel
from model.vgg_model import VGG11, VGG13, VGG16, VGG19
from model.efficientnet_model import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, \
    EfficientNetB5, EfficientNetB6, EfficientNetB7


class ModelLoader:
    def __init__(self, device):
        self.device = device
        self.models_dict = {
            # 'meddef': MedDef,
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
            # 'resnext101_32x8d': ResNeXt101_32x8d,
            # 'resnext101_64x4d': ResNeXt101_64x4d,
            # 'vgg11': VGG11,
            # 'vgg13': VGG13,
            # 'vgg16': VGG16,
            # 'vgg19': VGG19,
            # 'efficientnet_b0': EfficientNetB0,
            # 'efficientnet_b1': EfficientNetB1,
            # 'efficientnet_b2': EfficientNetB2,
            # 'efficientnet_b3': EfficientNetB3,
            # 'efficientnet_b4': EfficientNetB4,
            # 'efficientnet_b5': EfficientNetB5,
            # 'efficientnet_b6': EfficientNetB6,
            # 'efficientnet_b7': EfficientNetB7,
            # 'mobilenet_v2': MobileNetV2Model,
            # 'mobilenet_v3_small': MobileNetV3SmallModel,
            # 'hybrid_resnet_densenet': HybridResNetDenseNet,
        }
        logging.info("ModelLoader initialized with models: " + ", ".join(self.models_dict.keys()))

    def get_model(self, model_name, input_channels=3, num_classes=None, pretrained=True):
        if pretrained:
            logging.info(
                f"Getting pretrained model {model_name} with {input_channels} input channels "
                f"and {num_classes} output classes.")
        else:
            logging.info(
                f"Getting model {model_name} with {input_channels} input channels "
                f"and {num_classes} output classes, pretrained={pretrained}.")

        if model_name not in self.models_dict:
            raise ValueError(f"Model {model_name} not recognized.")

        if num_classes is None:
            raise ValueError("num_classes must be specified")

        model_class = self.models_dict[model_name]
        model = model_class(pretrained=pretrained, input_channels=input_channels, num_classes=num_classes)

        if torch.cuda.is_available():
            model = model.to(self.device)
        return model

    def load_pretrained_model(self, model_name, load_task, dataset_name, input_channels=3, num_classes=None):
        if model_name not in self.models_dict:
            raise ValueError(f"Model {model_name} not recognized.")

        if num_classes is None:
            raise ValueError("num_classes must be specified")

        model_class = self.models_dict[model_name]
        model = model_class(pretrained=False, input_channels=input_channels, num_classes=num_classes)
        model = model.to(self.device)

        model_path = f"out/{load_task}/{dataset_name}/save_model/best_{model_name}_{dataset_name}.pth"
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise ValueError(f"No saved model found at {model_path}")

        return model
import os
import torch
import logging

from model.backbone.cbam.resnet_cbam import get_resnet_with_cbam
from model.backbone.resnet import get_resnet
from model.densenet_model import get_densenet
from model.meddef.meddef import get_meddef
from model.vgg_model import get_vgg
from model.attention.MSARNet import MSARNet


class ModelLoader:
    def __init__(self, device, arch, pretrained=True):
        self.device = device
        self.arch = arch
        self.pretrained = pretrained

        # Define model architectures and their depths
        self.models_dict = {
            'resnet': {'func': get_resnet, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes']},
            'densenet': {'func': get_densenet, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes']},
            'vgg': {'func': get_vgg, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes']},
            'msarnet': {'func': MSARNet, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes']},
            'cbam_resnet': {'func': get_resnet_with_cbam, 'params': ['depth', 'input_channels', 'num_classes', 'robust_method']},
            'meddef': {'func': get_meddef, 'params': ['depth', 'input_channels', 'num_classes', 'robust_method']}

        }
        logging.info("ModelLoader initialized with models: " + ", ".join(self.models_dict.keys()))

    def get_model(self, model_name=None, depth=None, input_channels=3, num_classes=None):
        """Retrieves a model based on specified architecture, depth, and configurations."""
        model_name = model_name or self.arch

        if model_name not in self.models_dict:
            raise ValueError(f"Model {model_name} not recognized.")

        if num_classes is None:
            raise ValueError("num_classes must be specified")

        model_entry = self.models_dict[model_name]
        model_func = model_entry['func']
        model_params = model_entry['params']

        # Prepare the arguments for the model function
        kwargs = {
            'depth': depth,
            'pretrained': self.pretrained,
            'input_channels': input_channels,
            'num_classes': num_classes
        }

        # Filter the kwargs to only include the required parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_params}

        if isinstance(depth, list):
            models = {}
            for d in depth:
                filtered_kwargs['depth'] = d
                model = model_func(**filtered_kwargs)
                model_name_with_depth = f"{model_name}{d}" if 'depth' in model_params else model_name
                if torch.cuda.is_available():
                    model = model.to(self.device)
                models[d] = (model, model_name_with_depth)
            return models
        else:
            model = model_func(**filtered_kwargs)
            model_name_with_depth = f"{model_name}{depth}" if 'depth' in model_params else model_name
            if torch.cuda.is_available():
                model = model.to(self.device)
            return model, model_name_with_depth

    def load_pretrained_model(self, model_name, load_task, dataset_name, depth=None, input_channels=3, num_classes=None):
        """Loads a pretrained model, specified by architecture, depth, and task-related information."""
        model, model_name_with_depth = self.get_model(
            model_name=model_name,
            depth=depth,
            input_channels=input_channels,
            num_classes=num_classes
        )

        model = model.to(self.device)

        model_path = f"out/{load_task}/{dataset_name}/save_model/best_{model_name_with_depth}_{dataset_name}.pth"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)  # for multi-GPU use

        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f"Loaded pretrained model from {model_path}")
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}")

        return model

    def load_multiple_models(self, model_name, depths, input_channels=3, num_classes=None):
        """Loads multiple models of the same architecture but different depths, as specified."""
        models = {}
        for depth in depths:
            try:
                model, model_name_with_depth = self.get_model(
                    model_name=model_name,
                    depth=depth,
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                models[depth] = model
                logging.info(f"Model {model_name_with_depth} loaded successfully.")
            except ValueError as e:
                logging.error(f"Failed to load model {model_name} with depth {depth}: {e}")
        return models
# model_loader.py

import gc
import os
import torch
import logging

from model.backbone.cbam.resnet_cbam import get_resnet_with_cbam
from model.backbone.resnet import get_resnet
from model.densenet_model import get_densenet
from model.meddef.meddef import get_meddef
from model.vgg_model import get_vgg
from model.attention.MSARNet import MSARNet
from model.attention.self_resnet import get_resnetsa

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
            'meddef': {'func': get_meddef, 'params': ['depth', 'input_channels', 'num_classes', 'robust_method']},
            'resnetsa': {'func': get_resnetsa, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes']}
        }
        logging.info("ModelLoader initialized with models: " + ", ".join(self.models_dict.keys()))

    def get_latest_checkpoint(self, model_name_with_depth, dataset_name, load_task):
        """Finds the most recent checkpoint for the given model and dataset."""
        checkpoint_dir = f"out/{load_task}/{dataset_name}/{model_name_with_depth}/save_model"
        if not os.path.exists(checkpoint_dir):
            print(f"ModelLoader: No checkpoint directory found for {model_name_with_depth} in {checkpoint_dir}")
            return None

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"best_{model_name_with_depth}_{dataset_name}")]
        if not checkpoints:
            print(f"ModelLoader: No checkpoints found for {model_name_with_depth} in {checkpoint_dir}")
            return None

        # Sort checkpoints by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        latest_checkpoint = checkpoints[0]
        return os.path.join(checkpoint_dir, latest_checkpoint)

    # model_loader.py

    def get_model(self, model_name=None, depth=None, input_channels=3, num_classes=None, task_name=None,
                  dataset_name=None):
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

        model_name_with_depth = f"{model_name}{depth}" if 'depth' in model_params else model_name

        # Check if a checkpoint exists
        model = None
        if task_name and dataset_name:
            checkpoint_path = self.get_latest_checkpoint(model_name_with_depth, dataset_name, task_name)
            if checkpoint_path:
                # Load the checkpoint to CPU first to avoid OOM issues
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model = model_func(**filtered_kwargs)
                model.load_state_dict(checkpoint)
                logging.info(f"Loaded pretrained model from checkpoint: {checkpoint_path}")
                # Delete the checkpoint variable and free any cached memory
                del checkpoint
                gc.collect()
                torch.cuda.empty_cache()
                # Now move the model to the specified device
                model = model.to(self.device)
            else:
                logging.info(f"No checkpoint found for {model_name_with_depth}. Creating a new model.")

        # If model is still None, it means no checkpoint was found, so create a new model
        if model is None:
            model = model_func(**filtered_kwargs)
            logging.info(f"ModelLoader: Created a new model: {model_name_with_depth}")

        # Move the model to the specified device
        if torch.cuda.is_available():
            model = model.to(self.device)

        return model, model_name_with_depth



    def load_pretrained_model(self, model_name, load_task, dataset_name, depth=None, input_channels=3, num_classes=None):
        """Loads a pretrained model, specified by architecture, depth, and task-related information."""
        model, model_name_with_depth = self.get_model(
            model_name=model_name,
            depth=depth,
            input_channels=input_channels,
            num_classes=num_classes,
            task_name=load_task,
            dataset_name=dataset_name
        )

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)  # for multi-GPU use

        return model

    def load_multiple_models(self, model_name, depths, input_channels=3, num_classes=None, task_name=None, dataset_name=None):
        """Loads multiple models of the same architecture but different depths, as specified."""
        models = {}
        for depth in depths:
            try:
                model, model_name_with_depth = self.get_model(
                    model_name=model_name,
                    depth=depth,
                    input_channels=input_channels,
                    num_classes=num_classes,
                    task_name=task_name,
                    dataset_name=dataset_name
                )
                models[depth] = model
                logging.info(f"Model {model_name_with_depth} loaded successfully.")
            except ValueError as e:
                logging.error(f"Failed to load model {model_name} with depth {depth}: {e}")
        return models
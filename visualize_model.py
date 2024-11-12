# visualize_model.py

import torch
from torch import nn
from torchviz import make_dot

def visualize_model(model, input_tensor, model_name):
    y = model(input_tensor)
    make_dot(y, params=dict(model.named_parameters())).render(f"out/{model_name}_graph", format="png")

def visualize_all_models(args, models_dict):
    for model_name in args.arch:
        depths = args.depth.get(model_name, []) if isinstance(args.depth, dict) else [args.depth]
        for depth in depths:
            if isinstance(depth, list):
                depth = tuple(depth)  # Convert list to tuple to make it hashable
            model_info = models_dict.models_dict.get(model_name)
            if isinstance(model_info, dict):
                model, model_name_with_depth = model_info.get(depth, (None, None))
            else:
                model, model_name_with_depth = model_info, model_name

            if isinstance(model, nn.Module):  # Ensure model is an instance of nn.Module
                input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
                visualize_model(model, input_tensor, model_name_with_depth)
                print(f"Model {model_name_with_depth} visualized.")
            else:
                print(f"Skipping {model_name_with_depth} as it is not a model instance.")
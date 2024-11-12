# visualize_model.py is a script that visualizes the computation graph of a PyTorch model using the torchviz library. It loads a dataset and model, and then generates a random input tensor to pass through the model. The script then visualizes the computation graph of the model and saves it as a PNG file.

import torch
import torchvision
from torch import nn
from torchviz import make_dot
import logging
import random
import os
import json
from loader.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from utils.logger import setup_logger
from utils.robustness.optimizers import OptimizerLoader
from utils.robustness.lr_scheduler import LRSchedulerLoader
from arg_parser import get_args


def setup_environment(args):
    print("Torch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)
    print("CUDA available: ", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device ID: {i}")
        print(f"  Name: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
        print(f"  Is Integrated: {props.is_integrated}")
        print(f"  Is Multi-GPU Board: {props.is_multi_gpu_board}")
        for attr in dir(props):
            if not attr.startswith('_'):
                print(f"  {attr}: {getattr(props, attr)}")
        print()

    setup_logger('out/logger.txt')
    logging.info("Visualize script started.")

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    if args.device.type == 'cuda' and args.device.index is not None:
        device_index = args.device_index
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is out of range. Available devices: {torch.cuda.device_count()}")


def initialize_components(args):
    datasets_dict = DatasetLoader.get_all_datasets(args.data, args.data_dir)
    models_dict = ModelLoader(args.device, args.arch, args.pretrained)
    optimizers_dict = OptimizerLoader()
    lr_scheduler_loader = LRSchedulerLoader()
    return datasets_dict, models_dict, optimizers_dict, lr_scheduler_loader


def visualize_model(model, input_tensor, model_name):
    y = model(input_tensor)
    make_dot(y, params=dict(model.named_parameters())).render(f"out/{model_name}_graph", format="png")


def visualize_all_models(args, models_dict, input_channels_dict, num_classes):
    for model_name in args.arch:
        depths = args.depth.get(model_name, []) if isinstance(args.depth, dict) else [args.depth]
        for depth in depths:
            if isinstance(depth, list):
                depth = tuple(depth)
            try:
                input_channels = input_channels_dict.get(args.data[0])
                model, model_name_with_depth = models_dict.get_model(
                    model_name, depth=depth, input_channels=input_channels, num_classes=num_classes
                )
                model.to(args.device)  # Move model to the specified device
            except ValueError as e:
                print(f"Failed to load model {model_name} with depth {depth}: {e}")
                continue

            if isinstance(model, nn.Module):
                input_tensor = torch.randn(1, 3, 224, 224).to(args.device)  # Move input tensor to the same device
                visualize_model(model, input_tensor, model_name_with_depth)
                print(f"Model {model_name_with_depth} visualized.")
            else:
                print(f"Skipping {model_name_with_depth} as it is not a model instance.")


if __name__ == "__main__":
    args = get_args()
    setup_environment(args)
    datasets_dict, models_dict, optimizers_dict, lr_scheduler_loader = initialize_components(args)

    input_channels_dict = {dataset_name: dataset_loader.get_input_channels(
        train_batch_size=args.train_batch,
        val_batch_size=args.test_batch,
        test_batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    ) for dataset_name, dataset_loader in datasets_dict.items()}

    first_dataset_loader = next(iter(datasets_dict.values()))
    train_loader, _, _ = first_dataset_loader.load(
        train_batch_size=args.train_batch,
        val_batch_size=args.test_batch,
        test_batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    num_classes = DatasetLoader.get_dataloader_target_class_number(train_loader)[1]

    visualize_all_models(args, models_dict, input_channels_dict, num_classes)
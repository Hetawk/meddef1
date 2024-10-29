# main.py

import logging
import random
import sys

import torch
import torchvision
from torch import nn

from loader.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from utils.logger import setup_logger
from utils.robustness.optimizers import OptimizerLoader
from utils.task_handler import TaskHandler
from utils.robustness.lr_scheduler import LRSchedulerLoader
from utils.robustness.cross_validation import CrossValidator  # Import CrossValidator
from arg_parser import get_args

print("Torch version: ", torch.__version__)
print("Torchvision version: ", torchvision.__version__)
print("CUDA available: ", torch.cuda.is_available())
print('System Version:', sys.version)
print('System Version Information:', sys.version_info)
print('System Path:', sys.path)
print('System Executable:', sys.executable)
print('System Platform:', sys.platform)
print('System Maxsize:', sys.maxsize)
print('System Implementation:', sys.implementation)

# Set up logging
setup_logger('out/logger.txt')
logging.info("Main script started.")

# Parse arguments
args = get_args()
state = {k: v for k, v in args._get_kwargs()}

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manualSeed)

# Use the get_all_datasets static method from the DatasetLoader class to get the dictionary of all datasets
datasets_dict = DatasetLoader.get_all_datasets()

# Check if a specific dataset is provided in the command line arguments
if args.data:
    datasets_dict = {args.data: datasets_dict[args.data]}

models_dict = ModelLoader(args.device, args.arch)  # Pass device argument when creating ModelLoader instance
optimizers_dict = OptimizerLoader()
lr_scheduler_loader = LRSchedulerLoader()  # Initialize LRSchedulerLoader

# Define a single set of hyperparameters to be used for all datasets
hyperparams = {
    'epochs': args.epochs,
    'lr': args.lr,
    'momentum': args.momentum,
    'patience': args.patience,
    'lambda_l2': args.lambda_l2,
    'dropout_rate': args.drop,
    'optimizer': 'adam',
    'batch_size': args.train_batch,
    'scheduler': 'StepLR',  # Add scheduler name
    'scheduler_params': {'step_size': 10, 'gamma': args.gamma}  # Add scheduler parameters
}

# record to be used for tuning
results = []

# Iterate over each dataset in datasets_dict
for dataset_name, dataset_loader in datasets_dict.items():
    dataset_loader.pin_memory = args.pin_memory  # Set pin_memory for each dataset_loader
    input_channels = dataset_loader.get_input_channels(
        train_batch_size=args.train_batch,
        val_batch_size=args.test_batch,
        test_batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )

    # Load datasets
    train_loader, val_loader, test_loader = dataset_loader.load(
        train_batch_size=args.train_batch,
        val_batch_size=args.test_batch,
        test_batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )

    # Extract train_dataset from train_loader
    train_dataset = train_loader.dataset
    # Check if train_dataset is a Subset and get the original dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        original_dataset = train_dataset.dataset
    else:
        original_dataset = train_dataset
    # Get the classes from the original dataset
    classes = original_dataset.classes
    num_classes = len(classes)

    # Iterate over each model in models_dict
    for model_name, model_class in models_dict.models_dict.items():
        for depth in args.depth:
            cross_validator = CrossValidator(
                dataset=train_loader.dataset,
                model=models_dict.models_dict[model_name],
                model_name=model_name,
                dataset_name=dataset_name,
                criterion=nn.CrossEntropyLoss(),
                optimizer_class=optimizers_dict.optimizers_dict[hyperparams['optimizer']],
                optimizer_params={'lr': hyperparams['lr'], 'momentum': hyperparams['momentum']},
                hyperparams=hyperparams,
                num_classes=num_classes,
                device=args.device,
                args=args,
                attack_loader=None,
                scheduler=None,
                cross_validator=None
            )

            task_handler = TaskHandler(
                datasets_dict={dataset_name: dataset_loader},  # Use only the specified dataset
                models_loader=models_dict,
                optimizers_dict=optimizers_dict,
                hyperparams_dict={dataset_name: hyperparams},
                input_channels_dict={dataset_name: input_channels},
                classes={dataset_name: classes},
                dataset_name=dataset_name,  # Pass the specified dataset name
                lr_scheduler_loader=lr_scheduler_loader,
                cross_validator=cross_validator,
                device=args.device,
                args=args
            )

            task_handler.args.depth = [depth]  # Ensure depth is always a list
            if args.task_name == 'normal_training':
                task_handler.run_train()
            elif args.task_name == 'attack':
                task_handler.run_attack()
            elif args.task_name == 'defense':
                task_handler.run_defense()
            else:
                logging.error(f"Unknown task: {args.task_name}. No task was executed.")

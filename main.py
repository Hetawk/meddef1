# main.py

import logging
import torch
import torchvision
from loader.dataset_loader import DatasetLoader
from model.base_model import ModelLoader
from utils.logger import setup_logger
from utils.robustness.cross_validation import CrossValidator
from utils.robustness.optimizers import OptimizerLoader
from utils.helper import TaskHandler
from utils.robustness.lr_scheduler import LRSchedulerLoader

print("Torch version: ", torch.__version__)
print("Torchvision version: ", torchvision.__version__)
print("CUDA available: ", torch.cuda.is_available())
# Set up logging
setup_logger('out/logger.txt')
logging.info("Main script started.")

# Use the get_all_datasets static method from the DatasetLoader class to get the dictionary of all datasets
datasets_dict = DatasetLoader.get_all_datasets()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Determine device based on GPU availability -> linux server
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Determine device based on GPU availability -> using local computer with gpu
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine device based on GPU availability

models_dict = ModelLoader(device)  # Pass device argument when creating ModelLoader instance
optimizers_dict = OptimizerLoader()
lr_scheduler_loader = LRSchedulerLoader()  # Initialize LRSchedulerLoader

# Define a single set of hyperparameters to be used for all datasets
hyperparams = {
    'epochs': 2,
    'lr': 0.001,
    'momentum': 0.9,
    'patience': 5,
    'lambda_l2': 0.01,
    'dropout_rate': 0.5,
    'optimizer': 'adam',
    'batch_size': 32,
}

task_to_run = (  # Uncomment the task that you wish to run
    'train'
    # 'attack'
    # 'defense'
)
# Iterate over each dataset in datasets_dict
for dataset_name, dataset_loader in datasets_dict.items():
    datasets = dataset_loader.load()
    train_dataset, val_dataset, test_dataset = datasets[:3]
    classes = datasets[-1]
    input_channels = dataset_loader.get_input_channels()

    # Initialize task_handler outside the model iteration
    task_handler = TaskHandler(
        {dataset_name: dataset_loader},
        models_dict,
        optimizers_dict,
        {dataset_name: hyperparams},
        {dataset_name: input_channels},
        classes,
        dataset_name,
        lr_scheduler_loader,
        None,  # Placeholder for CrossValidator parameters
        device
    )

    if task_to_run == 'train':
        task_handler.run_train()
    elif task_to_run == 'attack':
        task_handler.run_attack()
    elif task_to_run == 'defense':
        task_handler.run_defense()

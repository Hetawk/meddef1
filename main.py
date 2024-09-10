# main.py
import json
import logging
import sys

import torch
import torchvision
from torch import nn

# from arg_parser import ArgParser
from loader.dataset_loader import DatasetLoader
from loader.preprocess import Preprocessor
from model.model_loader import ModelLoader
from train import Trainer
from utils.logger import setup_logger
from utils.robustness.optimizers import OptimizerLoader
from utils.task_handler import TaskHandler
from utils.robustness.lr_scheduler import LRSchedulerLoader
from utils.robustness.cross_validation import CrossValidator  # Import CrossValidator
from utils.evaluator import Evaluator

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
# arg_parser = ArgParser()
# args = arg_parser.get_args()
# logging.info(f"Parsed arguments: {args}")

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
    'epochs': 30,
    # 'epochs': args.epochs,
    'lr': 0.01,
    # 'lr': args.lr,
    'momentum': 0.9,
    'patience': 5,
    'lambda_l2': 0.01,
    'dropout_rate': 0.5,
    'optimizer': 'adam',
    'batch_size': 32,
    # 'batch_size': args.batch_size,
    'scheduler': 'StepLR',  # Add scheduler name
    'scheduler_params': {'step_size': 10, 'gamma': 0.1}  # Add scheduler parameters
}

task_to_run = (  # Uncomment the task that you wish to run
    'normal_training'
    # 'attack'
    # 'defense'
)

# record to be used for tuning
results = []

# Iterate over each dataset in datasets_dict
for dataset_name, dataset_loader in datasets_dict.items():
    input_channels = dataset_loader.get_input_channels()

    # Load datasets
    datasets = dataset_loader.load()
    train_dataset, val_dataset, test_dataset = datasets[:3] if len(datasets) > 2 else (datasets[0], None, None)

    # Iterate over each model in models_dict
    for model_name in models_dict.models_dict.keys():
        # Initialize Preprocessor
        preprocessor = Preprocessor(
            model_type=model_name,
            dataset_name=dataset_name,
            task_name=task_to_run,
            data_dir='./dataset',
            hyperparams=hyperparams
        )

        # Preprocess datasets once
        train_dataset, val_dataset, test_dataset = preprocessor.preprocess(
            train_dataset, val_dataset, test_dataset, input_channels
        )

        classes = preprocessor.extract_classes(train_dataset)
        num_classes = len(classes)

        # After preprocessing the datasets
        train_loader, val_loader, test_loader = preprocessor.wrap_datasets_in_dataloaders(
            train_dataset, val_dataset, test_dataset, shuffle=True
        )

        # Initialize CrossValidator
        cross_validator = CrossValidator(
            dataset=train_dataset,
            model=models_dict.models_dict[model_name],
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizers_dict.get_optimizer(
                hyperparams['optimizer'],
                model_params=models_dict.models_dict[model_name](num_classes=num_classes).parameters(),
                lr=hyperparams['lr'],
                momentum=hyperparams['momentum']
            ),
            hyperparams=hyperparams
        )

        # Initialize TaskHandler with preprocessed datasets
        task_handler = TaskHandler(
            datasets_dict={dataset_name: (train_loader, val_loader, test_loader)},
            models_loader=models_dict,
            optimizers_dict=optimizers_dict,
            hyperparams_dict={dataset_name: hyperparams},
            input_channels_dict={dataset_name: input_channels},
            classes={dataset_name: classes},  # Pass classes as a dictionary
            dataset_name=dataset_name,
            lr_scheduler_loader=lr_scheduler_loader,  # Pass lr_scheduler_loader
            cross_validator=cross_validator,  # Pass cross_validator
            device=device,
            preprocessor=preprocessor
        )

        if task_to_run == 'normal_training':
            task_handler.run_train()
        elif task_to_run == 'attack':
            task_handler.run_attack()
        elif task_to_run == 'defense':
            task_handler.run_defense()
        else:
            logging.error(f"Unknown task: {task_to_run}. No task was executed.")
#             continue
#
#         trainer = Trainer(
#             model=models_dict.get_model(model_name, input_channels=input_channels, num_classes=num_classes),
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             optimizer='adam',
#             criterion=nn.CrossEntropyLoss(),
#             model_name=model_name,
#             task_name=task_to_run,
#             dataset_name=dataset_name,
#             device=device,
#             lambda_l2=0.0,
#             dropout_rate=0.5,
#             alpha=0.01,
#             attack_loader=None,
#             scheduler=None,
#             cross_validator=None,
#             adversarial=False
#         )
#
#         # Retrieve test results
#         true_labels, all_predictions = trainer.get_test_results()
#         task_name = task_to_run
#         all_probabilities = []  # Assuming you have a way to get probabilities if needed
#
#         evaluator = Evaluator(model_name, results, true_labels.tolist(), all_predictions.tolist(), task_name,
#                               all_probabilities)
#         metrics = evaluator.evaluate(dataset_name)
#
#         # Filter the desired metrics
#         filtered_metrics = {
#             'Accuracy': metrics['accuracy'],
#             'Precision': metrics['precision'],
#             'Recall': metrics['recall'],
#             'F1 Score': metrics['f1']
#         }
#
#         # Record results
#         result = {
#             'dataset': dataset_name,
#             'model': model_name,
#             'task': task_to_run,
#             'hyperparameters': hyperparams,
#             'transformations': preprocessor.get_transforms(input_channels, dataset_name),
#             'performance': filtered_metrics
#         }
#         results.append(result)
# # Save results to a JSON file
# with open('tuning.json', 'w') as f:
#     json.dump(results, f, indent=4)
#
# logging.info("Script finished.")
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
from utils.robustness.cross_validation import CrossValidator
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
        # List all available attributes
        for attr in dir(props):
            if not attr.startswith('_'):
                print(f"  {attr}: {getattr(props, attr)}")
        print()

    # Set up logging
    setup_logger('out/logger.txt')
    logging.info("Main script started.")

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    # Check if the specified CUDA device is available
    if args.device.type == 'cuda' and args.device.index is not None:
        device_index = args.device_index
        if device_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device_index} is out of range. Available devices: {torch.cuda.device_count()}")

def get_hyperparams(args):
    return {
        'epochs': args.epochs,
        'lr': args.lr,
        'momentum': args.momentum,
        'patience': args.patience,
        'lambda_l2': args.lambda_l2,
        'dropout_rate': args.drop,
        'optimizer': 'adam',
        'batch_size': args.train_batch,
        'scheduler': 'StepLR',
        'scheduler_params': {'step_size': 10, 'gamma': args.gamma}
    }

def initialize_components(args):
    datasets_dict = DatasetLoader.get_all_datasets(args.data, args.data_dir)
    models_dict = ModelLoader(args.device, args.arch, args.pretrained)
    optimizers_dict = OptimizerLoader()
    lr_scheduler_loader = LRSchedulerLoader()
    return datasets_dict, models_dict, optimizers_dict, lr_scheduler_loader

def process_dataset(dataset_name, dataset_loader, args, models_dict, optimizers_dict, lr_scheduler_loader, hyperparams):
    dataset_loader.pin_memory = args.pin_memory
    input_channels = dataset_loader.get_input_channels(
        train_batch_size=args.train_batch,
        val_batch_size=args.test_batch,
        test_batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )

    train_loader, val_loader, test_loader = dataset_loader.load(
        train_batch_size=args.train_batch,
        val_batch_size=args.test_batch,
        test_batch_size=args.test_batch,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )

    train_dataset = train_loader.dataset
    original_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    classes, num_classes = DatasetLoader.get_dataloader_target_class_number(train_loader)

    for model_name in args.arch:
        if isinstance(args.depth, dict):
            depths = args.depth.get(model_name, [])
            if not depths:
                logging.warning(f"No depths specified for model {model_name}. Using default depth.")
                depths = [None]
        elif isinstance(args.depth, list):
            depths = args.depth
        else:
            depths = [args.depth]

        logging.info(f"Using depths for model {model_name}: {depths}")
        for depth in depths:
            logging.info(f"Loading model with depth: {depth} for dataset: {dataset_name} and task: {args.task_name}")
            model, model_name_with_depth = models_dict.get_model(
                model_name=model_name,
                depth=depth,
                input_channels=3,
                num_classes=num_classes,
                task_name=args.task_name,
                dataset_name=dataset_name
            )
            logging.info(f"Model {model_name_with_depth} loaded successfully")

            cross_validator = CrossValidator(
                dataset=train_loader.dataset,
                model=model,
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
                datasets_dict={dataset_name: dataset_loader},
                models_loader=models_dict,
                optimizers_dict=optimizers_dict,
                hyperparams_dict={dataset_name: hyperparams},
                input_channels_dict={dataset_name: input_channels},
                classes={dataset_name: classes},
                dataset_name=dataset_name,
                lr_scheduler_loader=lr_scheduler_loader,
                cross_validator=cross_validator,
                device=args.device,
                args=args,
                num_classes=num_classes
            )

            task_handler.args.depth = [depth]
            if args.task_name == 'normal_training':
                task_handler.run_train()
            elif args.task_name == 'attack':
                task_handler.run_attack()
            elif args.task_name == 'defense':
                task_handler.run_defense()
            else:
                logging.error(f"Unknown task: {args.task_name}. No task was executed.")

def main():
    args = get_args()
    setup_environment(args)
    hyperparams = get_hyperparams(args)
    datasets_dict, models_dict, optimizers_dict, lr_scheduler_loader = initialize_components(args)

    for dataset_name in args.data:
        dataset_loader = datasets_dict[dataset_name]
        for arch in args.arch:
            models_dict.arch = arch
            process_dataset(dataset_name, dataset_loader, args, models_dict, optimizers_dict, lr_scheduler_loader, hyperparams)

if __name__ == "__main__":
    main()
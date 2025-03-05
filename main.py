import utils.pandas_patch
import logging
import os
import torch
import torchvision
import torch.multiprocessing as mp
from utils.logger import setup_logger
from utils.task_handler import TaskHandler
from train import parse_args  # Changed this line - import from train.py instead


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


def main():
    # Basic setup
    if os.name == 'nt':
        mp.set_start_method('spawn')
    else:
        mp.set_start_method('forkserver')

    args = parse_args()
    setup_environment(args)
    torch.cuda.empty_cache()

    # Initialize TaskHandler with all tasks
    task_handler = TaskHandler(args)

    # Execute task based on task_name
    if args.task_name == 'normal_training':
        task_handler.run_train()
    elif args.task_name == 'attack':
        task_handler.run_attack()
    elif args.task_name == 'defense':
        task_handler.run_defense()
    else:
        logging.error(f"Unknown task: {args.task_name}. No task was executed.")


if __name__ == "__main__":
    main()

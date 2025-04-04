import warnings
from argument_parser import parse_args
from utils.task_handler import TaskHandler
from utils.logger import setup_logger
import torch.multiprocessing as mp
import torchvision
import torch
import os
import logging
import utils.pandas_patch
from utils.matplotlib_config import configure_matplotlib_backend
from utils.file_handler import FileHandler

# Configure matplotlib backend before any other imports that might use matplotlib
configure_matplotlib_backend()

# Suppress ALL warnings right from the start
warnings.filterwarnings("ignore")

# Apply visualization patch
try:
    from utils.visual.visualization_patch import patch_visualization
    patch_visualization()
except:
    pass

# Suppress common warnings
FileHandler.suppress_warnings()


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

    # Set up logging with model-specific path when available
    # For task_name, we always have this
    # For dataset_name and model_name, we might have multiple or none, so handle appropriately
    task_name = args.task_name

    # Get the first dataset and model name when available
    dataset_name = args.data[0] if hasattr(
        args, 'data') and args.data else None

    # For model name, combine arch and depth when available
    model_name = None
    if hasattr(args, 'arch') and args.arch and hasattr(args, 'depth'):
        arch = args.arch[0]  # First architecture
        # Get depth string representation for path
        if isinstance(args.depth, dict) and arch in args.depth:
            depth_value = str(args.depth[arch][0]) if args.depth[arch] else ""
            model_name = f"{arch}_{depth_value}" if depth_value else arch

    # Initialize the logger without timestamps
    setup_logger(task_name=task_name, dataset_name=dataset_name,
                 model_name=model_name)
    logging.info("Main script started.")


def main():
    # Basic setup
    if os.name == 'nt':
        mp.set_start_method('spawn')
    else:
        mp.set_start_method('forkserver')

    args = parse_args()  # Remove the mode parameter
    setup_environment(args)
    torch.cuda.empty_cache()

    # Initialize TaskHandler with all tasks
    task_handler = TaskHandler(args)

    # Execute task based on task_name
    if args.task_name == 'normal_training':
        # Add quick_test_after_training flag to arguments
        run_test = getattr(args, 'quick_test_after_training', False)
        task_handler.run_train(run_test)
    elif args.task_name == 'attack':
        task_handler.run_attack()
    elif args.task_name == 'defense':
        task_handler.run_defense()
    else:
        logging.error(f"Unknown task: {args.task_name}. No task was executed.")


if __name__ == "__main__":
    main()

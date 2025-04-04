import argparse
import json
import os
import warnings
import logging  # Use Python's standard logging module instead

import torch

# Suppressing FutureWarnings that might come from argument parsing
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    """Parse command line arguments with a unified argument set"""
    parser = argparse.ArgumentParser(
        description='MedDef Training/Testing Configuration')

    # Dataset and processing arguments
    parser.add_argument('--data', nargs='+', required=True, type=str,
                        help='Dataset names to process')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory')

    # Model architecture arguments
    parser.add_argument('--arch', '-a', nargs='+', default=['meddef1', 'resnet', 'densenet'],
                        help='Architecture(s) to use. Provide one or multiple values. Separate multiple names with space or comma.')
    parser.add_argument('--depth', type=str, default='{"meddef1": [1.0, 1.1], "resnet": [18, 34], "densenet": [121]}',
                        help='Model depths as JSON string')

    # Device configuration
    parser.add_argument('--gpu-ids', default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--device-index', type=int, default=0,
                        help='Primary GPU index to use')

    # Task specification
    parser.add_argument('--task_name', type=str,
                        choices=['normal_training', 'attack', 'defense'],
                        default='normal_training',
                        help='Task to perform')

    # Seed for reproducibility
    parser.add_argument('--manualSeed', type=int, default=None,
                        help='manual seed for reproducibility')

    # Training parameters
    parser.add_argument('--train_batch', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=32,
                        help='Test/Val batch size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    # Loss function options
    parser.add_argument('--loss_type', type=str, default='standard',
                        choices=['standard', 'weighted',
                                 'aggressive', 'dynamic'],
                        help='Type of loss function to use for handling class imbalance')
    parser.add_argument('--focal_alpha', type=float, default=0.5,
                        help='Alpha parameter for focal loss in dynamic sample weighting')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Base gamma parameter for focal loss in dynamic sample weighting')

    # Early stopping configuration
    parser.add_argument('--patience', type=int, default=25,
                        help='Patience for early stopping (max 25)')
    parser.add_argument('--min_epochs', type=int, default=0,
                        help='Minimum epochs to train regardless of early stopping')
    parser.add_argument('--early_stopping_metric', type=str, default='loss',
                        choices=['loss', 'accuracy', 'f1', 'balanced_acc'],
                        help='Metric to monitor for early stopping')

    # Regularization parameters
    parser.add_argument('--drop', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lambda_l2', type=float, default=1e-4,
                        help='L2 regularization strength')

    # Training control
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm')

    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adagrad'],
                        help='Optimizer to use for training')

    # Scheduler options
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'ExponentialLR', 'ReduceLROnPlateau',
                                 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_step', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')

    # Adversarial options - unified for both training and testing
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable adversarial training/testing')
    parser.add_argument('--attack_type', type=str, nargs='+',
                        default=['fgsm'],
                        choices=['fgsm', 'pgd', 'bim', 'jsma'],
                        help='Type(s) of attack. Can specify multiple attacks.')
    parser.add_argument('--attack_eps', type=float, default=0.1,
                        help='Epsilon for adversarial attacks')
    parser.add_argument('--initial_epsilon', type=float, default=None,
                        help='Initial epsilon for progressive adversarial training')
    parser.add_argument('--epsilon_steps', type=int, default=5,
                        help='Number of epochs to increase epsilon over')
    parser.add_argument('--adv_weight', type=float, default=0.5,
                        help='Final weight for adversarial loss')
    parser.add_argument('--initial_adv_weight', type=float, default=0.2,
                        help='Initial weight for adversarial loss')
    parser.add_argument('--attack_alpha', type=float, default=0.01,
                        help='Step size for iterative attacks')
    parser.add_argument('--attack_steps', type=int, default=10,
                        help='Number of steps for iterative attacks')
    parser.add_argument('--adv_init_mag', type=float, default=0.01,
                        help='Initial magnitude for adversarial perturbation')
    parser.add_argument('--save_attacks', action='store_true',
                        help='Save generated adversarial samples')

    # Adversarial training specific arguments
    parser.add_argument('--dynamic_alpha', type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Dynamically adjust alpha based on epsilon (default: True)')
    parser.add_argument('--epsilon_schedule', type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Schedule for epsilon progression (default: linear)')
    parser.add_argument('--combined_early_stopping', type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Use combined clean and adversarial metrics for early stopping')
    parser.add_argument('--adaptive_schedule', action='store_true',
                        help='Use adaptive three-phase training schedule')

    # ANFIS options
    parser.add_argument('--use_anfis', action='store_true',
                        help='Enable ANFIS for adversarial training')
    parser.add_argument('--anfis_sensitivity_weight', type=float, default=0.7,
                        help='Weight for sensitivity component in ANFIS')
    parser.add_argument('--anfis_perturbation_weight', type=float, default=0.3,
                        help='Weight for perturbation magnitude component in ANFIS')

    # Defense task options
    parser.add_argument('--prune_rate', type=float, default=0.3,
                        help='Pruning rate for defense task')

    # Testing specific arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model weights')
    parser.add_argument('--output_dir', type=str, default='out',
                        help='Base directory for outputs (default: out)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save model predictions')
    parser.add_argument('--evaluate_robustness', action='store_true',
                        help='Evaluate model robustness to adversarial attacks')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Confidence threshold for predictions')
    parser.add_argument('--fp16', action='store_true',
                        help='Use half precision for inference')
    parser.add_argument('--per_class_metrics', action='store_true',
                        help='Calculate per-class metrics')

    # Single image testing parameters
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a single image for testing')
    parser.add_argument('--show_heatmap', action='store_true',
                        help='Generate attention heatmap for visualization')

    # Final training option
    parser.add_argument('--quick_test_after_training', action='store_true',
                        help='Run a quick test evaluation after training completes')

    args = parser.parse_args()

    # Process architecture names
    if isinstance(args.arch, str):
        args.arch = [x.strip() for x in args.arch.strip('[]').split(',')]

    # Convert depth string to dictionary
    if isinstance(args.depth, str):
        # Remove spaces and single quotes
        depth_str = args.depth.strip()
        if depth_str.startswith("'") and depth_str.endswith("'"):
            depth_str = depth_str[1:-1]

        try:
            # First try direct JSON parsing
            args.depth = json.loads(depth_str)
        except json.JSONDecodeError:
            # If that fails, try manual parsing
            import ast
            try:
                # Remove curly braces
                depth_str = depth_str.strip('{}')
                # Split into key and value parts
                parts = depth_str.split(':', 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid depth format: {depth_str}")

                key = parts[0].strip().strip('"\'')
                value_str = parts[1].strip()

                # Try to parse the value as JSON array first
                try:
                    value = json.loads(value_str)
                except json.JSONDecodeError:
                    # If that fails, try to safely evaluate as Python literal
                    try:
                        value = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        # If all else fails, just treat it as a string list with quotes
                        value_str = value_str.strip('[]')
                        value = [v.strip().strip('"\'')
                                 for v in value_str.split(',')]

                args.depth = {key: value}
            except Exception as e:
                logging.error(f"Failed to parse depth argument: {e}")
                logging.error(
                    f"Please use valid JSON format like: --depth '{{\"transformer\": [\"tiny\"]}}'")
                raise

    # # Ensure patience doesn't exceed maximum
    # if hasattr(args, 'patience') and args.patience > 25:
    #     logging.warning(
    #         f"Patience value {args.patience} exceeds maximum (25). Setting to 25.")
    #     args.patience = 25

    # Configure CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    use_cuda = torch.cuda.is_available()
    args.device = torch.device(
        f"cuda:{args.device_index}" if use_cuda else "cpu")

    return args

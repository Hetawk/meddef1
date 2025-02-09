import argparse
import os
import torch
import json
import yaml


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Datasets
    parser.add_argument('-d', '--data', nargs='+', default=['ccts'], type=str)
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--data_dir', default='./dataset',
                        type=str, metavar='PATH', help='path to dataset')
    parser.add_argument('--use_cross_validator',
                        action='store_true', help='Use cross validation')
    parser.add_argument('--data_key', type=str, required=True,
                        help='Dataset name to use (e.g. scisic_train)')

    # Optimization options
    parser.add_argument('--epochs', default=3, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=2, type=int,
                        metavar='N', help='train batchsize (default: 2)')
    parser.add_argument('--test-batch', default=2, type=int,
                        metavar='N', help='test batchsize (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=0.01,
                        type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0,
                        type=float, metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--patience', default=5, type=int,
                        metavar='N', help='patience for early stopping')
    parser.add_argument('--lambda_l2', default=0.001, type=float,
                        metavar='L2', help='L2 regularization lambda')
    parser.add_argument('--accumulation_steps', default=16,
                        type=int, help='Number of gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float,
                        default=5.0, help='Gradient clipping norm')

    # Memory optimization options
    parser.add_argument('--cpu-offload', action='store_true', default=False,
                        help='Offload model to CPU when not in use')
    parser.add_argument('--optimize-memory', action='store_true', default=False,
                        help='Enable memory optimization techniques')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing to save memory')

    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # Architecture: now using nargs to accept one or more values
    parser.add_argument('--arch', '-a', nargs='+', default=['meddef', 'resnet', 'densenet'],
                        help='Architecture(s) to use. Provide one or multiple values. Separate multiple names with space or comma.')
    parser.add_argument('--depth', type=str, default='{"meddef": [1.0, 1.1], "resnet": [18, 34], "densenet": [121]}',
                        help='Model depths as a JSON string.')

    # Miscellaneous
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                        help='use pre-trained model')
    # Use pinned memory only if flag is specified; default is False
    parser.add_argument('--pin-memory', action='store_true',
                        default=False, help='Use pinned memory for data loading')

    # Add an FP16 flag to enable half precision (if desired)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use FP16 (half precision) for model loading/training')

    # Device options
    parser.add_argument('--gpu-ids', default='3,2,1,0',
                        type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--device-index', default=0, type=int,
                        help='CUDA device index (default: 0)')

    # Task to run
    parser.add_argument('--task_name', type=str, choices=['normal_training', 'attack', 'defense'],
                        default='normal_training', help='Task to run: normal_training, attack, or defense')

    # Attack options
    parser.add_argument('--attack_name', type=str, default='fgsm',
                        help='Name of the attack to use (e.g., fgsm, pgd)')
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='Epsilon value for the attack')
    parser.add_argument('--alpha', default=0.01, type=float, metavar='Alpha',
                        help='Alpha value for adversarial training')
    parser.add_argument('--iterations', default=40, type=int,
                        metavar='N', help='Number of iterations for the attack')

    # Defense options
    parser.add_argument('--prune_rate', type=float, default=0.2,
                        help='Pruning rate for unstructured pruning')

    parser.add_argument(
        '--config', default='./loader/config.yaml', help='Path to config file')

    args = parser.parse_args()

    # Process --arch: split each element by comma in case the user provided a comma-separated string
    new_arch = []
    for item in args.arch:
        new_arch.extend([x.strip() for x in item.strip("[]").split(",")])
    args.arch = new_arch

    # Load configuration file to override defaults
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # First get the dataset config
    datasets = config.get('data', {}).get('data_key', [])
    current_dataset = next(
        (ds for ds in datasets if ds['name'] == args.data[0]), None)

    if current_dataset:
        # Override with dataset-specific training config
        ds_training_cfg = current_dataset.get('training', {})
        if not args.evaluate:  # Only override if not in evaluation mode
            args.epochs = ds_training_cfg.get('epochs', args.epochs)
            args.lr = ds_training_cfg.get('lr', args.lr)
            args.weight_decay = ds_training_cfg.get(
                'weight_decay', args.weight_decay)
            args.momentum = ds_training_cfg.get('momentum', args.momentum)
            # Override batch size if not explicitly set in command line
            if args.train_batch == 2:  # default value
                args.train_batch = ds_training_cfg.get(
                    'batch_size', args.train_batch)

    # Ensure depth is always a dictionary
    try:
        args.depth = json.loads(args.depth.replace("'", "\""))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON format for depth argument: {args.depth}") from e

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    use_cuda = torch.cuda.is_available()
    args.device = torch.device(
        f"cuda:{args.device_index}" if use_cuda else "cpu")

    return args

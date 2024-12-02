import argparse
import os
import torch
import json

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Datasets
    parser.add_argument('-d', '--data', nargs='+', default=['ccts'], type=str)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--data_dir', default='./dataset', type=str, metavar='PATH', help='path to dataset')
    parser.add_argument('--use_cross_validator', action='store_true', help='Use cross validation')

    # Optimization options
    parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=64, type=int, metavar='N', help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=32, type=int, metavar='N', help='test batchsize (default: 200)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--patience', default=5, type=int, metavar='N', help='patience for early stopping')
    parser.add_argument('--lambda_l2', default=0.01, type=float, metavar='L2', help='L2 regularization lambda')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Number of gradient accumulation steps')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # Architecture
    parser.add_argument('--arch', '-a', nargs='+', metavar='ARCH', default=['msarnet', 'resnet', 'densenet'])
    parser.add_argument('--depth', type=str, default='{"msarnet": [18, 34], "resnet": [18, 34], "densenet": [121]}',
                        help='Model depths as a JSON string.')

    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--pin-memory', action='store_false', help='Use pinned memory for data loading')

    # Device options
    parser.add_argument('--gpu-ids', default='3,2,1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--device-index', default=0, type=int, help='CUDA device index (default: 3)')

    # Task to run
    parser.add_argument('--task_name', type=str, choices=['normal_training', 'attack', 'defense'],
                        default='normal_training', help='Task to run: normal_training, attack, or defense')

    # Attack options
    parser.add_argument('--attack_name', type=str, default='fgsm', help='Name of the attack to use (e.g., fgsm, pgd)')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Epsilon value for the attack')
    parser.add_argument('--alpha', default=0.01, type=float, metavar='Alpha',
                        help='Alpha value for adversarial training')
    parser.add_argument('--iterations', default=40, type=int, metavar='N', help='Number of iterations for the attack')

    # Defense options
    parser.add_argument('--prune_rate', type=float, default=0.2, help='Pruning rate for unstructured pruning')

    args = parser.parse_args()

    # Ensure depth is always a dictionary
    args.depth = json.loads(args.depth)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    use_cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda:{args.device_index}" if use_cuda else "cpu")

    return args
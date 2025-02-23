# optimizer.py

import torch.optim as optim
import logging


class OptimizerLoader:
    """Handles optimizer initialization and loading"""

    def __init__(self):
        self.optimizers = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad
        }
        logging.info(
            f"OptimizerLoader initialized with optimizers: {', '.join(self.optimizers.keys())}")

    def get_optimizer(self, model, args):
        """Get optimizer based on model and arguments"""
        # Default to 'adam' if not specified
        optimizer_name = getattr(args, 'optimizer', 'adam').lower()

        if optimizer_name not in self.optimizers:
            logging.warning(
                f"Optimizer {optimizer_name} not found, using adam")
            optimizer_name = 'adam'

        optimizer_class = self.optimizers[optimizer_name]

        # Common parameters
        params = {
            'params': model.parameters(),
            'lr': args.lr
        }

        # Add optimizer-specific parameters
        if optimizer_name == 'sgd':
            params.update({
                'momentum': getattr(args, 'momentum', 0.9),
                'weight_decay': getattr(args, 'weight_decay', 1e-4),
                'nesterov': getattr(args, 'nesterov', False)
            })
        elif optimizer_name == 'adam':
            params.update({
                'weight_decay': getattr(args, 'weight_decay', 1e-4),
                'betas': getattr(args, 'betas', (0.9, 0.999))
            })

        optimizer = optimizer_class(**params)
        logging.info(
            f"Created {optimizer_name} optimizer with learning rate {args.lr}")
        return optimizer

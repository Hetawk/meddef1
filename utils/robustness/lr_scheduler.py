# lr_scheduler.py

import torch.optim.lr_scheduler as lr_scheduler
import logging


class LRSchedulerLoader:
    """
    Loads appropriate learning rate scheduler based on configuration
    """

    def __init__(self):
        """Initialize the scheduler loader"""
        logging.info("LRSchedulerLoader initialized with schedulers: StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR")

    def get_scheduler(self, optimizer, scheduler='StepLR', config=None):
        """
        Get scheduler based on name

        Args:
            optimizer: PyTorch optimizer
            scheduler: Name of scheduler (StepLR, ExponentialLR, etc.)
            config: Configuration object with scheduler parameters

        Returns:
            PyTorch scheduler
        """
        if config is None:
            config = {}

        # Extract parameters with defaults
        step_size = getattr(config, 'lr_step', 30)
        gamma = getattr(config, 'lr_gamma', 0.1)
        patience = getattr(config, 'lr_patience', 10)
        min_lr = getattr(config, 'min_lr', 1e-6)
        t_max = getattr(config, 'epochs', 100)

        scheduler = scheduler.lower() if isinstance(scheduler, str) else 'steplr'

        if scheduler == 'steplr':
            return lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler == 'exponentiallr':
            return lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        elif scheduler == 'reducelronplateau':
            return lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=gamma,
                patience=patience, min_lr=min_lr, verbose=True
            )
        elif scheduler == 'cosineannealinglr':
            return lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=min_lr
            )
        elif scheduler == 'cosineannealingwarmrestarts':
            return lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=step_size, T_mult=1, eta_min=min_lr
            )
        elif scheduler == 'onecyclelr':
            steps_per_epoch = getattr(config, 'steps_per_epoch', 100)
            return lr_scheduler.OneCycleLR(
                optimizer, max_lr=getattr(config, 'lr', 0.001) * 10,
                steps_per_epoch=steps_per_epoch, epochs=t_max
            )
        else:
            logging.warning(
                f"Unknown scheduler {scheduler}, not using any scheduler")
            return None

# For backward compatibility


def get_scheduler(optimizer, scheduler='StepLR', config=None):
    """Function version of LRSchedulerLoader.get_scheduler for backward compatibility"""
    loader = LRSchedulerLoader()
    return loader.get_scheduler(optimizer, scheduler, config)

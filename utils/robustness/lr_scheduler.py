# lr_scheduler.py

import torch.optim.lr_scheduler as lr_scheduler
import logging


class LRSchedulerLoader:
    def __init__(self):
        self.schedulers = {
            'StepLR': lr_scheduler.StepLR,
            'ExponentialLR': lr_scheduler.ExponentialLR,
            'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
        }
        logging.info(
            f"LRSchedulerLoader initialized with schedulers: {', '.join(self.schedulers.keys())}")

    def get_scheduler(self, optimizer, args):
        # Get scheduler name from args or use default
        scheduler_name = getattr(args, 'scheduler', 'StepLR')

        if scheduler_name not in self.schedulers:
            logging.warning(
                f"Scheduler {scheduler_name} not found, using StepLR")
            scheduler_name = 'StepLR'

        scheduler_class = self.schedulers[scheduler_name]

        # Configure scheduler based on type
        if scheduler_name == 'StepLR':
            return scheduler_class(
                optimizer,
                step_size=getattr(args, 'lr_step', 30),
                gamma=getattr(args, 'lr_gamma', 0.1)
            )
        elif scheduler_name == 'ExponentialLR':
            return scheduler_class(
                optimizer,
                gamma=getattr(args, 'lr_gamma', 0.95)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            return scheduler_class(
                optimizer,
                mode='min',
                factor=getattr(args, 'lr_gamma', 0.1),
                patience=getattr(args, 'lr_patience', 10),
                verbose=True
            )

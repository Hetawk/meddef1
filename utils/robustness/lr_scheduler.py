# lr_scheduler.py

import torch.optim.lr_scheduler as lr_scheduler
import logging

class LRSchedulerLoader:
    def __init__(self):
        self.schedulers_dict = {
            'StepLR': lr_scheduler.StepLR,
            'ExponentialLR': lr_scheduler.ExponentialLR,
            'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
        }
        self.logger = logging.getLogger(__name__)
        self.logger.info("LRSchedulerLoader initialized with schedulers: " + ", ".join(self.schedulers_dict.keys()))

    def get_scheduler(self, scheduler_name, optimizer, **kwargs):
        if scheduler_name in self.schedulers_dict:
            self.logger.info(f"Loading LR scheduler: {scheduler_name} with params: {kwargs}")
            return self.schedulers_dict[scheduler_name](optimizer, **kwargs)
        else:
            raise ValueError(f"LR Scheduler {scheduler_name} not recognized.")


# optimizer.py

import torch.optim as optim
import logging

class OptimizerLoader:
    def __init__(self):
        self.optimizers_dict = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad
        }
        logging.info("OptimizerLoader initialized with optimizers: " + ", ".join(self.optimizers_dict.keys()))

    def get_optimizer(self, optimizer_name, model_params, **kwargs):
        if optimizer_name in self.optimizers_dict:
            logging.info(f"Loading optimizer: {optimizer_name} with params: {kwargs}")
            if optimizer_name == 'adam':
                # Use 'momentum' value for the first beta coefficient if it's specified
                betas = (kwargs.pop('momentum', 0.9), 0.999)
                return self.optimizers_dict[optimizer_name](model_params, betas=betas, **kwargs)
            else:
                return self.optimizers_dict[optimizer_name](model_params, **kwargs)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not recognized.")

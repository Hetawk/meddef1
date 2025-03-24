# optimizer.py

import torch.optim as optim
import logging


class OptimizerLoader:
    """
    Loads appropriate optimizer based on configuration
    """

    def __init__(self):
        """Initialize the optimizer loader"""
        logging.info(
            "OptimizerLoader initialized with optimizers: sgd, adam, rmsprop, adagrad")

    def get_optimizer(self, model, optimizer_name='adam', lr=0.001, weight_decay=1e-4, **kwargs):
        """
        Get optimizer based on name

        Args:
            model: PyTorch model
            optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop', 'adagrad')
            lr: Learning rate
            weight_decay: Weight decay factor
            **kwargs: Additional optimizer-specific parameters

        Returns:
            PyTorch optimizer
        """
        optimizer_name = optimizer_name.lower()

        # Filter only parameters that require gradients
        parameters = [p for p in model.parameters() if p.requires_grad]

        if optimizer_name == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            nesterov = kwargs.get('nesterov', True)
            optimizer = optim.SGD(parameters, lr=lr, momentum=momentum,
                                  weight_decay=weight_decay, nesterov=nesterov)
            logging.info(f"Created SGD optimizer with learning rate {lr}")
        elif optimizer_name == 'adam':
            betas = kwargs.get('betas', (0.9, 0.999))
            optimizer = optim.Adam(parameters, lr=lr, betas=betas,
                                   weight_decay=weight_decay)
            logging.info(f"Created adam optimizer with learning rate {lr}")
        elif optimizer_name == 'rmsprop':
            alpha = kwargs.get('alpha', 0.99)
            momentum = kwargs.get('momentum', 0)
            optimizer = optim.RMSprop(parameters, lr=lr, alpha=alpha,
                                      momentum=momentum, weight_decay=weight_decay)
            logging.info(f"Created RMSprop optimizer with learning rate {lr}")
        elif optimizer_name == 'adagrad':
            optimizer = optim.Adagrad(
                parameters, lr=lr, weight_decay=weight_decay)
            logging.info(f"Created Adagrad optimizer with learning rate {lr}")
        else:
            logging.warning(
                f"Unknown optimizer {optimizer_name}, defaulting to Adam")
            optimizer = optim.Adam(
                parameters, lr=lr, weight_decay=weight_decay)

        return optimizer

    def get_param_groups(self, model):
        """
        Create parameter groups for differential learning rates
        """
        # This is a placeholder for implementing more complex parameter grouping strategies
        return [{'params': model.parameters()}]

# For backward compatibility


def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4, **kwargs):
    """Function version of OptimizerLoader.get_optimizer for backward compatibility"""
    loader = OptimizerLoader()
    return loader.get_optimizer(model, optimizer_name, lr, weight_decay, **kwargs)

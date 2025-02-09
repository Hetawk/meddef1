# regularization.py

import torch
import torch.nn as nn
import logging


class Regularization:
    """
    A class for applying various regularization techniques to a model.
    """

    @staticmethod
    def apply_l2_regularization(model, lambda_l2, log_message=True):
        """
        Applies L2 regularization to the given model.

        Args:
            model (nn.Module): The model to apply L2 regularization to.
            lambda_l2 (float): The regularization strength.

        Returns:
            torch.Tensor: The L2 regularization term to be added to the loss.
            :param model:
            :param lambda_l2:
            :param log_message:
        """
        if log_message:
            print("\n")
            logging.info(
                f"Applying L2 regularization with strength: {lambda_l2}")
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # Ensure lambda_l2 is consistent with your configuration (e.g. 0.001)
        return lambda_l2 * l2_norm

    @staticmethod
    def apply_dropout(model, dropout_rate):
        """
        Applies dropout to the given model by updating the dropout rate.

        Args:
            model (nn.Module): The model to apply dropout to.
            dropout_rate (float): The dropout rate.

        Returns:
            None
        """
        logging.info(f"Applying dropout with rate: {dropout_rate}")
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    @staticmethod
    def integrate_regularization(loss, l2_reg, log_message=True):
        """
        Integrates L2 regularization term into the loss.

        Args:
            loss (torch.Tensor): The original loss.
            l2_reg (torch.Tensor): The L2 regularization term.

        Returns:
            torch.Tensor: The combined loss with L2 regularization.
        """
        if log_message:
            logging.info(f"Integrating L2 regularization into the loss.")
        return loss + l2_reg

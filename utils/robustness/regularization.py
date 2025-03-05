# regularization.py

import torch
import torch.nn as nn
import logging


class Regularization:
    """
    A class for applying various regularization techniques to a model.
    """

    @staticmethod
    def apply_l2_regularization(model, lambda_l2, log_message=False):
        """Calculate L2 regularization loss"""
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2)
        l2_loss = lambda_l2 * l2_loss

        if log_message:
            logging.info(
                f"Applying L2 regularization with strength: {lambda_l2}")

        return l2_loss

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
    def integrate_regularization(loss, l2_reg, log_message=False):
        """Add regularization to the loss"""
        if log_message:
            logging.info("Integrating L2 regularization into the loss.")
        return loss + l2_reg

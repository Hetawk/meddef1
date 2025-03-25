import torch
import torch.nn as nn
import logging


class Attack:
    """
    Base class for all adversarial attacks.

    This serves as the foundation for implementing various attack methods
    like FGSM, PGD, BIM, etc.
    """

    def __init__(self, model, config):
        """
        Initialize attack with model and configuration

        Args:
            model: The model to attack
            config: Configuration parameters
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        # Default loss function for adversarial attacks
        self.loss_fn = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        """
        Base attack method. Should be overridden by subclasses.

        Args:
            images: Input images to be attacked
            labels: True labels for the images

        Returns:
            Tuple of (original images, adversarial images, additional info)
        """
        raise NotImplementedError("Subclasses must implement attack method")

    def generate(self, images, labels, **kwargs):
        """
        Generate adversarial examples. Should be overridden by subclasses.

        Args:
            images: Input images
            labels: True labels
            **kwargs: Additional attack parameters

        Returns:
            Adversarial examples
        """
        raise NotImplementedError("Subclasses must implement generate method")

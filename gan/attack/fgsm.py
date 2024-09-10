# fgsm.py

import torch
import torch.nn as nn
import logging

class FGSMAttack:
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.

    This attack perturbs the input image by adjusting each pixel based on the
    sign of the gradient of the loss with respect to the input. It can be used
    for both targeted and non-targeted attacks.

    Attributes:
        model (torch.nn.Module): The target model to be attacked.
        epsilon (float): The perturbation magnitude.
        targeted (bool): Indicates if the attack is targeted or non-targeted.
        device (torch.device): The device on which the model and inputs are located.

    Methods:
        attack(images, labels):
            Performs the FGSM attack on a batch of images and returns the original and perturbed images.
        denorm(batch, mean, std):
            Denormalizes a batch of tensors to their original scale.
    """

    def __init__(self, model, epsilon, targeted=False):
        """
        Initializes the FGSM attack with the given model, epsilon, and attack type.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            epsilon (float): The perturbation magnitude.
            targeted (bool, optional): Indicates if the attack is targeted or non-targeted. Default is False.
        """
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.device = next(model.parameters()).device
        logging.info("FGSM Attack initialized.")

    def attack(self, images, labels):
        """
        Performs the FGSM attack on a batch of images and returns the original and perturbed images.

        Args:
            images (torch.Tensor): Batch of input images.
            labels (torch.Tensor): True labels for the images.

        Returns:
            tuple: A tuple containing:
                - original_images (torch.Tensor): The original images.
                - perturbed_images (torch.Tensor): The perturbed (adversarial) images.
                - labels (torch.Tensor): The true labels (if provided).
        """
        logging.info("Performing FGSM attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        images.requires_grad = True

        outputs = self.model(images)
        self.model.zero_grad()

        if self.targeted:
            cost = -loss(outputs, labels)  # Targeted attack: maximize loss on the true label
        else:
            cost = loss(outputs, labels)   # Non-targeted attack: maximize loss on predicted label

        cost.backward()

        # Collect the element-wise sign of the data gradient
        gradient_sign = images.grad.data.sign()

        # Create perturbed image by adjusting each pixel based on the sign of the gradient
        perturbed_images = images + self.epsilon * gradient_sign

        # Clamp perturbed images to ensure valid pixel range [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        if labels is not None:
            return images.detach(), perturbed_images.detach(), labels
        else:
            return images.detach(), perturbed_images.detach()

    # def denorm(self, batch, mean=None, std=None):
    #     """
    #     Denormalizes a batch of tensors to their original scale.
    #
    #     Args:
    #         batch (torch.Tensor): Batch of normalized tensors.
    #         mean (torch.Tensor or list): Mean used for normalization.
    #         std (torch.Tensor or list): Standard deviation used for normalization.
    #
    #     Returns:
    #         torch.Tensor: Batch of tensors without normalization applied to them.
    #     """
    #     if mean is None:
    #         mean = [0.1307]
    #     if std is None:
    #         std = [0.3081]
    #
    #     if isinstance(mean, list):
    #         mean = torch.tensor(mean).to(self.device)
    #     if isinstance(std, list):
    #         std = torch.tensor(std).to(self.device)
    #
    #     return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


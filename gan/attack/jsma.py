# jsma.py

import torch
import logging

class JSMAAttack:
    """
    Implements the Jacobian-based Saliency Map Attack (JSMA).

    This attack perturbs the input image by adjusting pixels based on the saliency map, which is derived from
    the gradients of the target model's output with respect to the input. The goal is to misclassify the input
    by perturbing the least number of pixels.

    Attributes:
        model (torch.nn.Module): The target model to be attacked.
        theta (float): The number of iterations for the attack.
        gamma (float): The perturbation magnitude per pixel.
        clip_min (float): Minimum value for clipping the pixel values.
        clip_max (float): Maximum value for clipping the pixel values.
        device (torch.device): The device on which the model and inputs are located.

    Methods:
        attack(images, labels):
            Performs the JSMA attack on a batch of images and returns the perturbed images.
        _perturb_image(images, sign_grad):
            Perturbs the images based on the sign of the gradient.
        denorm(batch, mean=[0.1307], std=[0.3081]):
            Convert a batch of tensors to their original scale.
    """

    def __init__(self, model, theta=1.0, gamma=0.1, clip_min=0.0, clip_max=1.0):
        """
        Initializes the JSMA attack with the given model, theta, gamma, and clipping values.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            theta (float, optional): The number of iterations for the attack. Default is 1.0.
            gamma (float, optional): The perturbation magnitude per pixel. Default is 0.1.
            clip_min (float, optional): Minimum value for clipping the pixel values. Default is 0.0.
            clip_max (float, optional): Maximum value for clipping the pixel values. Default is 1.0.
        """
        self.model = model
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = next(model.parameters()).device
        logging.info("JSMA Attack initialized.")

    def attack(self, images, labels):
        """
        Performs the JSMA attack on a batch of images and returns the perturbed images.

        Args:
            images (torch.Tensor): Batch of input images.
            labels (torch.Tensor): True labels for the images.

        Returns:
            torch.Tensor: The perturbed (adversarial) images.
        """
        logging.info("Performing JSMA attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)

        perturbed_images = images.clone().detach().requires_grad_(True)

        for _ in range(self.theta):
            # Forward pass
            outputs = self.model(perturbed_images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            # Create saliency map
            grad = perturbed_images.grad.data.clone()
            sign_grad = torch.sign(grad)

            # Perturb the image pixels
            perturbed_images = self._perturb_image(perturbed_images, sign_grad)

            # Clip the perturbed image to ensure pixel values are within [clip_min, clip_max]
            perturbed_images = torch.clamp(perturbed_images, self.clip_min, self.clip_max)

        return perturbed_images

    def _perturb_image(self, images, sign_grad):
        """
        Perturbs the images based on the sign of the gradient.

        Args:
            images (torch.Tensor): Batch of input images.
            sign_grad (torch.Tensor): Sign of the gradient of the loss with respect to the input images.

        Returns:
            torch.Tensor: The perturbed (adversarial) images.
        """
        # Create the adversarial perturbation
        perturbation = torch.zeros_like(images).to(self.device)

        for i in range(images.size(0)):
            perturbation[i] += sign_grad[i]

        perturbed_images = images + self.gamma * perturbation
        return perturbed_images

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


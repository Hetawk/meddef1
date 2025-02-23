# jsma.py

import torch
import logging


class JSMAAttack:
    """
    Implements the Jacobian-based Saliency Map Attack (JSMA).

    This attack perturbs the input image by adjusting pixels based on the saliency map,
    derived from the gradients of the model output with respect to the input.
    """

    def __init__(self, model, theta=1.0, gamma=0.1, clip_min=0.0, clip_max=1.0):
        self.model = model
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._logged = False
        self.device = next(model.parameters()).device
        logging.info("JSMA Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing JSMA attack.")
            self._logged = True
        images = images.to(self.device)
        labels = labels.to(self.device)
        # Clone original images to return later
        original_images = images.clone().detach()
        # Clone and require gradients for perturbation
        perturbed_images = images.clone().detach().requires_grad_(True)
        for _ in range(int(self.theta)):
            outputs = self.model(perturbed_images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            grad = perturbed_images.grad.data.clone()
            sign_grad = torch.sign(grad)
            # Perturb the images using gamma as the step size
            perturbed_images = self._perturb_image(perturbed_images, sign_grad)
            perturbed_images = torch.clamp(
                perturbed_images, self.clip_min, self.clip_max)
            # Prepare for next iteration
            perturbed_images = perturbed_images.detach().requires_grad_(True)
        # Return original images, adversarial images, and labels as a tuple
        return original_images, perturbed_images.detach(), labels.detach()

    def generate(self, images, labels, epsilon=None):
        # Call attack() and ignore the original images and labels
        _, perturbed_images, _ = self.attack(images, labels)
        return perturbed_images

    def _perturb_image(self, images, sign_grad):
        # Perturb images by adding gamma-scaled sign of gradients
        return images + self.gamma * sign_grad

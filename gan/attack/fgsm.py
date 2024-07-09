# fgsm.py

import torch
import torch.nn as nn
import logging

class FGSMAttack:
    def __init__(self, model, epsilon, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.device = next(model.parameters()).device
        logging.info("FGSM Attack initialized.")

    def attack(self, images, labels):
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

        # Ensure the number of adversarial examples matches the batch size
        if len(perturbed_images) < len(images):
            diff = len(images) - len(perturbed_images)
            perturbed_images = torch.cat((perturbed_images, images[-diff:]))

        if labels is not None:
            return images.detach(), perturbed_images.detach(), labels
        else:
            return images.detach(), perturbed_images.detach()

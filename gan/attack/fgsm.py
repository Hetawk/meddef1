import torch
import torch.nn as nn
import logging

class FGSMAttack:
    def __init__(self, model, epsilon=0.3, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.device = next(model.parameters()).device
        logging.info("FGSM Attack initialized.")

    def attack(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        self.model.zero_grad()

        cost = -loss(outputs, labels) if self.targeted else loss(outputs, labels)
        cost.backward()

        gradient_sign = images.grad.data.sign()
        perturbed_images = images + self.epsilon * gradient_sign
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return images.detach(), perturbed_images.detach(), labels.detach()

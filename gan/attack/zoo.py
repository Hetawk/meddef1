import torch
import torch.nn.functional as F
import logging
import numpy as np


class ZooAttack:
    def __init__(self, model, epsilon, num_iterations=10, alpha=0.01, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.targeted = targeted
        self.device = next(model.parameters()).device
        logging.info("Zoo Attack initialized.")

    def attack(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        orig_outputs = self.model(images)
        orig_preds = orig_outputs.argmax(dim=1)

        adversarial_images = images.clone().detach()
        adversarial_images.requires_grad = True

        for _ in range(self.num_iterations):
            outputs = self.model(adversarial_images)
            loss = self._loss_fn(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            grad_sign = adversarial_images.grad.sign()
            perturbed_images = adversarial_images + self.alpha * grad_sign
            perturbed_images = torch.clamp(perturbed_images, min=0, max=1)

            if self.targeted:
                loss = -self._loss_fn(outputs, orig_preds)
            else:
                loss = self._loss_fn(outputs, labels)

            adversarial_images = perturbed_images.detach()
            adversarial_images.requires_grad = True

            if torch.norm(adversarial_images - images, p=np.inf) > self.epsilon:
                delta = (adversarial_images - images).clamp(-self.epsilon, self.epsilon)
                adversarial_images = (images + delta).clamp(0, 1).detach()
                adversarial_images.requires_grad = True

        return adversarial_images

    def _loss_fn(self, outputs, labels):
        if self.targeted:
            return -F.cross_entropy(outputs, labels)
        else:
            return F.cross_entropy(outputs, labels)

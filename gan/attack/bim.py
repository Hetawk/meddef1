import torch
import torch.nn as nn
import logging


class BIMAttack:
    def __init__(self, model, epsilon, alpha, iterations):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.device = next(model.parameters()).device
        self._logged = False  # flag to log message only once
        logging.info("BIM Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing BIM attack.")
            self._logged = True
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss_func = nn.CrossEntropyLoss()

        perturbed_images = images.clone().detach()
        perturbed_images.requires_grad = True

        for _ in range(self.iterations):
            outputs = self.model(perturbed_images)
            cost = loss_func(outputs, labels)
            grad = torch.autograd.grad(cost, perturbed_images)[0]
            perturbed_images = perturbed_images + self.alpha * torch.sign(grad)
            # Clip the perturbed images to be within epsilon ball and valid image range
            perturbed_images = torch.max(
                torch.min(perturbed_images, images + self.epsilon), images - self.epsilon)
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            # Detach and set requires_grad to True for the next iteration
            perturbed_images = perturbed_images.detach()
            perturbed_images.requires_grad = True

        return images.detach(), perturbed_images.detach(), labels.detach()

    def generate(self, images, labels, epsilon):
        # Use the attack method but return only the adversarial samples
        _, adv_data, _ = self.attack(images, labels)
        return adv_data

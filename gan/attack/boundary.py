import torch
import logging

class BoundaryAttack:
    def __init__(self, model, epsilon, iterations):
        self.model = model
        self.epsilon = epsilon
        self.iterations = iterations
        self.device = next(model.parameters()).device
        logging.info("Boundary Attack initialized.")

    def attack(self, images, labels):
        logging.info("Performing Boundary attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = torch.nn.CrossEntropyLoss()

        perturbed_images = images.clone().detach()
        perturbed_images.requires_grad = True

        for _ in range(self.iterations):
            outputs = self.model(perturbed_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, perturbed_images)[0]

            # Perturb image in the direction of gradient sign
            perturbation = self.epsilon * torch.sign(grad)
            perturbed_images = perturbed_images + perturbation

            # Clamp perturbed images to ensure pixel values stay within [0, 1] range
            perturbed_images = torch.clamp(perturbed_images, min=0, max=1).detach_()

        return perturbed_images

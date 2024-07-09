import torch
import logging

class ElasticNetAttack:
    def __init__(self, model, epsilon, alpha, iterations, beta=1.0):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.beta = beta
        self.device = next(model.parameters()).device
        logging.info("ElasticNet Attack initialized.")

    def attack(self, images, labels):
        logging.info("Performing ElasticNet attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = torch.nn.CrossEntropyLoss()

        perturbed_images = images.clone().detach()
        perturbed_images.requires_grad = True

        for _ in range(self.iterations):
            outputs = self.model(perturbed_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, perturbed_images)[0]

            perturbation = self.epsilon * torch.sign(grad) + self.alpha * grad
            perturbed_images = perturbed_images + self.beta * perturbation
            perturbed_images = torch.clamp(perturbed_images, min=0, max=1).detach_()

        return perturbed_images

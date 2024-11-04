import torch
import logging

class BIMAttack:
    def __init__(self, model, epsilon, alpha, iterations):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.device = next(model.parameters()).device
        logging.info("BIM Attack initialized.")

    def attack(self, images, labels):
        logging.info("Performing BIM attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = torch.nn.CrossEntropyLoss()

        perturbed_images = images.clone().detach()
        perturbed_images.requires_grad = True

        for _ in range(self.iterations):
            outputs = self.model(perturbed_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, perturbed_images)[0]

            # Add perturbation scaled by alpha and the sign of the gradient
            perturbed_images = perturbed_images + self.alpha * torch.sign(grad)

            # Clip perturbations to ensure they are within epsilon neighborhood of original image
            perturbed_images = torch.max(torch.min(perturbed_images, images + self.epsilon), images - self.epsilon)

            # Clamp pixel values to [0, 1] range
            perturbed_images = torch.clamp(perturbed_images, min=0, max=1).detach_()

        return perturbed_images

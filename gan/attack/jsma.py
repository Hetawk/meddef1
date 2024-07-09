import torch
import logging

class JSMAAttack:
    def __init__(self, model, theta=1.0, gamma=0.1, clip_min=0.0, clip_max=1.0):
        self.model = model
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = next(model.parameters()).device
        logging.info("JSMA Attack initialized.")

    def attack(self, images, labels):
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
        # Create the adversarial perturbation
        perturbation = torch.zeros_like(images).to(self.device)

        for i in range(images.size(0)):
            perturbation[i] += sign_grad[i]

        perturbed_images = images + self.gamma * perturbation
        return perturbed_images

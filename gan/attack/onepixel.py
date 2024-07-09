import torch
import logging
import numpy as np
from scipy.optimize import differential_evolution

class OnePixelAttack:
    def __init__(self, model, pixel_count=1, max_iter=1000, popsize=400, epsilon=0.2):
        self.model = model
        self.pixel_count = pixel_count
        self.max_iter = max_iter
        self.popsize = popsize
        self.epsilon = epsilon
        self.device = next(model.parameters()).device
        logging.info("One Pixel Attack initialized.")

    def attack(self, images, labels):
        logging.info("Performing One Pixel attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)

        perturbed_images = images.clone().detach().requires_grad_(True)

        for i in range(batch_size):
            image = perturbed_images[i]

            # Define bounds for pixel positions (0 to image size - 1)
            bounds = [(0, image.size(1)-1), (0, image.size(2)-1)] * self.pixel_count

            # Perform differential evolution optimization to find optimal pixels
            result = differential_evolution(self._evaluate_pixel_attack, bounds,
                                            args=(image, labels[i].item()), maxiter=self.max_iter, popsize=self.popsize,
                                            tol=1e-3, mutation=(0.2, 0.8), recombination=1, seed=None, callback=None,
                                            disp=False, polish=True, init='latinhypercube', atol=0)

            # Apply the optimized attack pixels to the image
            pixels = np.rint(result.x).astype(int)
            for j in range(self.pixel_count):
                image[:, pixels[j*2], pixels[j*2+1]] = 1.0

            perturbed_images[i] = image

        return perturbed_images

    def _evaluate_pixel_attack(self, pixels, image, target_label):
        # Apply the attack pixels to the image
        pixels = np.rint(pixels).astype(int)
        for i in range(self.pixel_count):
            image[:, pixels[i*2], pixels[i*2+1]] = 1.0

        # Forward pass through the model and calculate the loss
        output = self.model(image.unsqueeze(0))
        loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target_label]).to(self.device))

        return loss.item()

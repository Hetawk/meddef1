# pgd.py

import torch
import torch.nn as nn
import logging

class PGDAttack:
    def __init__(self, model, epsilon, alpha, iterations):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self._logged = False  # flag to log message only once
        self.device = next(model.parameters()).device
        logging.info("PGD Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing PGD attack.")
            self._logged = True
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        perturbed_images = images.clone().detach()
        perturbed_images.requires_grad = True

        self.model.eval()  # Ensure model is in evaluation mode

        for _ in range(self.iterations):
            perturbed_images.grad = None  # Reset gradients
            outputs = self.model(perturbed_images)
            cost = loss(outputs, labels)
            cost.backward()

            with torch.no_grad():
                perturbed_images += self.alpha * perturbed_images.grad.sign()
                perturbed_images = torch.max(torch.min(perturbed_images, images + self.epsilon), images - self.epsilon)
                perturbed_images = torch.clamp(perturbed_images, 0, 1)

            perturbed_images.requires_grad = True  # Ensure gradients are calculated in the next iteration

        return images.detach(), perturbed_images.detach(), labels.detach()
        
    def generate(self, images, labels, epsilon=None):
        """Generate adversarial examples with optional epsilon override"""
        try:
            # Save original epsilon if we're overriding it
            if epsilon is not None:
                original_epsilon = self.epsilon
                self.epsilon = epsilon
                
            # Call attack and return only the adversarial samples
            _, perturbed_images, _ = self.attack(images, labels)
            
            # Restore original epsilon if it was overridden
            if epsilon is not None:
                self.epsilon = original_epsilon
                
            return perturbed_images
            
        except Exception as e:
            logging.error(f"Error in PGD generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails
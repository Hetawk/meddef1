import torch
import torch.nn as nn
import logging

class CWAttack:
    def __init__(self, model, confidence=0.0, learning_rate=0.01, binary_search_steps=9, max_iterations=1000, abort_early=True):
        self.model = model
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.device = next(model.parameters()).device
        logging.info("CW L2 Attack initialized.")

    def attack(self, images, labels):
        logging.info("Performing CW L2 attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)

        lower_bounds = torch.zeros_like(images)
        upper_bounds = torch.ones_like(images)

        perturbed_images = images.clone().detach().requires_grad_(True)

        return perturbed_images

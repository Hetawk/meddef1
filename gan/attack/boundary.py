import torch
import logging
import numpy as np

class BoundaryAttack:
    def __init__(self, model, initial_epsilon, step_size, iterations, targeted=False):
        """
        Initialize the Boundary Attack with specified parameters.

        :param model: The model to attack.
        :param initial_epsilon: Initial noise magnitude to start the perturbation.
        :param step_size: Step size for moving towards the decision boundary.
        :param iterations: Number of iterations for the attack.
        :param targeted: Boolean flag indicating whether the attack is targeted.
        """
        self.model = model
        self.initial_epsilon = initial_epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.targeted = targeted
        self.device = next(model.parameters()).device
        logging.info("Boundary Attack initialized.")

    def attack(self, images, labels):
        logging.info("Performing Boundary attack.")
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Start with a perturbed version of the image within initial_epsilon bounds
        perturbed_images = images + self.initial_epsilon * torch.randn_like(images).to(self.device)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Keep within valid range

        # Initial prediction check
        self.model.eval()
        original_predictions = self.model(images).argmax(dim=1)
        if self.targeted:
            attack_success = original_predictions == labels
        else:
            attack_success = original_predictions != labels

        # Iteratively refine the perturbation
        for i in range(self.iterations):
            # Generate a small random direction for perturbation
            random_direction = torch.randn_like(images).to(self.device)
            random_direction = random_direction / random_direction.view(random_direction.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)

            # Update perturbation
            candidate_images = perturbed_images + self.step_size * random_direction
            candidate_images = torch.clamp(candidate_images, 0, 1)

            # Check if the new image is closer to the boundary
            with torch.no_grad():
                candidate_predictions = self.model(candidate_images).argmax(dim=1)
                if self.targeted:
                    is_adversarial = candidate_predictions == labels
                else:
                    is_adversarial = candidate_predictions != labels

            # Update perturbed image if the candidate is adversarial and closer to target
            success_condition = is_adversarial if self.targeted else ~is_adversarial
            perturbed_images = torch.where(success_condition.to(torch.float32).view(-1, 1, 1, 1), candidate_images, perturbed_images)

            # Check if all images are successfully perturbed
            if success_condition.to(torch.float32).all():
                break

        return images.detach(), perturbed_images.detach(), labels.detach()
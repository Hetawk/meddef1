import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from gan.attack.attack import Attack


class PGDAttack(Attack):
    """
    PGD attack: Projected Gradient Descent

    Paper: "Towards Deep Learning Models Resistant to Adversarial Attacks"
    https://arxiv.org/abs/1706.06083
    """

    def __init__(self, model, config):
        super().__init__(model, config)
        self.eps = getattr(config, 'epsilon', 0.3)
        self.alpha = getattr(config, 'attack_alpha', 0.01)
        self.steps = getattr(config, 'attack_steps', 10)
        self.random_start = True

    def attack(self, images, labels):
        """
        Perform PGD attack on the given images
        """
        # Make copies to avoid modifying inputs
        original_images = images.clone().detach()
        labels = labels.clone().detach()

        # Save original model state
        training = self.model.training
        self.model.eval()

        # Initialize with original images
        adv_images = original_images.clone().detach()

        # Random start initialization if enabled
        if self.random_start:
            # Create random noise within epsilon constraints
            noise = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images + noise, 0, 1)
            # Ensure we're within epsilon ball
            delta = torch.clamp(
                adv_images - original_images, -self.eps, self.eps)
            adv_images = torch.clamp(original_images + delta, 0, 1)

        # Cache for best adversarial examples (most effective)
        best_adv_images = adv_images.clone().detach()
        best_loss = None

        # Perform attack steps
        for step in range(self.steps):
            # Ensure adv_images requires grad for this step
            adv_images = adv_images.detach()
            adv_images.requires_grad_(True)

            # Forward pass - make sure we're in a fresh computation context
            with torch.enable_grad():
                outputs = self.model(adv_images)
                # For an adversarial attack, we want to maximize the loss, not minimize it
                loss = -self.loss_fn(outputs, labels)

            # Compute gradients with proper error handling
            if adv_images.grad is not None:
                adv_images.grad.zero_()

            try:
                # Compute gradient of loss with respect to adv_images
                loss.backward()

                # Verify gradient exists
                if adv_images.grad is None:
                    logging.error("Gradient is None after backward pass")
                    break

                # Get the sign of the gradient for the attack
                grad_sign = adv_images.grad.sign()

                # Update adversarial examples
                adv_images = adv_images.detach() + self.alpha * grad_sign

                # Project back to epsilon ball and valid range
                delta = torch.clamp(
                    adv_images - original_images, -self.eps, self.eps)
                adv_images = torch.clamp(original_images + delta, 0, 1)

                # Check if these adversarial examples are better (higher loss)
                with torch.no_grad():
                    curr_outputs = self.model(adv_images)
                    curr_loss = self.loss_fn(curr_outputs, labels)

                    # If first step or new examples are better
                    if best_loss is None or curr_loss > best_loss:
                        best_loss = curr_loss
                        best_adv_images = adv_images.clone()

            except RuntimeError as e:
                logging.error(f"Backward pass failed: {e}")
                break

        # Restore model training state
        self.model.train(training)

        # Return the best adversarial examples found
        return original_images, best_adv_images.detach(), labels

    def generate(self, images, labels, epsilon=None):
        """
        Generate adversarial examples using PGD

        Args:
            images: Input images
            labels: True labels  
            epsilon: Attack strength (optional, overrides default)

        Returns:
            Adversarial examples
        """
        # Save original epsilon and restore later
        original_eps = self.eps
        original_alpha = self.alpha

        try:
            # Use provided epsilon if given
            if epsilon is not None:
                self.eps = epsilon
                # Adjust alpha based on epsilon to maintain proportionality
                if hasattr(self, 'dynamic_alpha') and self.dynamic_alpha:
                    self.alpha = min(epsilon * 0.25, 0.01)  # Cap alpha at 0.01
                else:
                    self.alpha = self.alpha  # Keep original alpha

            # Make sure images require gradients for attack
            x = images.clone()

            # Generate adversarial examples
            _, perturbed_images, _ = self.attack(x, labels)

            return perturbed_images
        except Exception as e:
            logging.error(f"Error in PGD generate: {e}")
            # Fall back to original images
            return images.clone()
        finally:
            # Restore original parameters
            self.eps = original_eps
            self.alpha = original_alpha

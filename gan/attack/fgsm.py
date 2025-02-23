import torch
import torch.nn as nn
import logging
import gc


class FGSMAttack:
    def __init__(self, model, epsilon, targeted=False):
        self.model = model
        self.epsilon = float(epsilon)
        self.targeted = targeted
        self.device = next(model.parameters()).device
        logging.info("FGSM Attack initialized.")

    def attack(self, images, labels):
        """Generate FGSM adversarial examples"""
        try:
            # Store original training mode and set to eval
            was_training = self.model.training
            self.model.eval()

            # Enable gradients for model parameters
            requires_grad_states = {}
            for name, param in self.model.named_parameters():
                requires_grad_states[name] = param.requires_grad
                param.requires_grad = True

            # Move data to appropriate device
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)
            images.requires_grad_(True)

            # Forward pass
            outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # If targeted, minimize loss for target class
            if self.targeted:
                loss = -loss

            # Backward pass
            loss.backward()

            # Create perturbation
            with torch.no_grad():
                gradient_sign = images.grad.sign()
                perturbed_images = images + self.epsilon * gradient_sign
                perturbed_images = torch.clamp(perturbed_images, 0, 1)

            # Cleanup and restore model states
            images.requires_grad_(False)
            for name, param in self.model.named_parameters():
                param.requires_grad = requires_grad_states[name]
            if was_training:
                self.model.train()

            # Memory cleanup
            del gradient_sign
            torch.cuda.empty_cache()
            gc.collect()

            return images.detach(), perturbed_images.detach(), labels.detach()

        except Exception as e:
            logging.error(f"Error in FGSM attack: {str(e)}", exc_info=True)
            # Return original images if attack fails
            return images.detach(), images.detach(), labels.detach()

    def generate(self, images, labels, epsilon=None):
        """Generate adversarial examples with optional epsilon override"""
        try:
            if epsilon is not None:
                original_epsilon = self.epsilon
                self.epsilon = epsilon

            _, perturbed_images, _ = self.attack(images, labels)

            if epsilon is not None:
                self.epsilon = original_epsilon

            return perturbed_images

        except Exception as e:
            logging.error(f"Error in FGSM generate: {str(e)}", exc_info=True)
            return images

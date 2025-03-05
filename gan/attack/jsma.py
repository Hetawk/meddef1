# jsma.py

import torch
import logging
import gc


class JSMAAttack:
    """
    Implements the Jacobian-based Saliency Map Attack (JSMA).

    This attack perturbs the input image by adjusting pixels based on the saliency map,
    derived from the gradients of the model output with respect to the input.
    """

    def __init__(self, model, epsilon=0.1, gamma=None, clip_min=0.0, clip_max=1.0):
        self.model = model
        self.epsilon = epsilon  # For consistency with other attacks, renamed theta to epsilon
        self.gamma = gamma if gamma is not None else epsilon/10  # Default gamma as epsilon/10
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._logged = False
        self.device = next(model.parameters()).device
        logging.info("JSMA Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing JSMA attack.")
            self._logged = True
            
        # Store original training mode and set to eval
        was_training = self.model.training
        self.model.eval()
            
        try:
            images = images.to(self.device)
            labels = labels.to(self.device)
            # Clone original images to return later
            original_images = images.clone().detach()
            # Clone and require gradients for perturbation
            perturbed_images = images.clone().detach().requires_grad_(True)
            
            for _ in range(int(1.0/self.gamma)):  # Scale iterations by gamma
                outputs = self.model(perturbed_images)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                self.model.zero_grad()
                loss.backward()
                grad = perturbed_images.grad.data.clone()
                sign_grad = torch.sign(grad)
                # Directly perturb the images using gamma as the step size
                perturbed_images = perturbed_images + self.gamma * sign_grad
                perturbed_images = torch.clamp(
                    perturbed_images, self.clip_min, self.clip_max)
                # Prepare for next iteration
                perturbed_images = perturbed_images.detach().requires_grad_(True)
                
            # Memory cleanup
            del grad, sign_grad
            torch.cuda.empty_cache()
            gc.collect()
            
            return original_images, perturbed_images.detach(), labels.detach()
            
        except Exception as e:
            logging.error(f"Error in JSMA attack: {str(e)}", exc_info=True)
            # Return original images if attack fails
            return images.detach(), images.detach(), labels.detach()
        finally:
            # Restore model's training mode
            if was_training:
                self.model.train()

    def generate(self, images, labels, epsilon=None):
        """Generate adversarial examples with optional epsilon override"""
        try:
            if epsilon is not None:
                original_epsilon = self.epsilon
                original_gamma = self.gamma
                self.epsilon = epsilon
                self.gamma = epsilon/10  # Adjust gamma proportionally
                
            # Call attack() and return only the adversarial images
            _, perturbed_images, _ = self.attack(images, labels)
            
            if epsilon is not None:
                self.epsilon = original_epsilon
                self.gamma = original_gamma
                
            return perturbed_images
            
        except Exception as e:
            logging.error(f"Error in JSMA generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails

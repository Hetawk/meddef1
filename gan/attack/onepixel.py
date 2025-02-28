import torch
import logging
import numpy as np
import gc
from scipy.optimize import differential_evolution

class OnePixelAttack:
    def __init__(self, model, epsilon=0.2, pixel_count=1, max_iter=100, popsize=10):
        """
        Initialize the One Pixel Attack.
        
        Args:
            model: The target model to attack
            epsilon: Controls attack strength (not directly used but kept for API consistency)
            pixel_count: Number of pixels to perturb
            max_iter: Maximum iterations for differential evolution
            popsize: Population size for differential evolution
        """
        self.model = model
        self.pixel_count = pixel_count
        self.max_iter = max_iter
        self.popsize = popsize
        self.epsilon = epsilon  # For consistency with other attacks
        self.device = next(model.parameters()).device
        self._logged = False
        logging.info("One Pixel Attack initialized.")

    def attack(self, images, labels):
        """Generate adversarial examples using One Pixel Attack"""
        if not self._logged:
            logging.info("Performing One Pixel attack.")
            self._logged = True
            
        # Store original training mode and set to eval
        was_training = self.model.training
        self.model.eval()
        
        try:
            images = images.to(self.device)
            labels = labels.to(self.device)
            # Clone original images to return later
            original_images = images.clone().detach()
            perturbed_images = images.clone().detach()

            batch_size = images.size(0)
            for i in range(batch_size):
                image = images[i].clone()
                label = labels[i].item()

                # Define bounds for pixel positions (0 to image size - 1)
                h, w = image.shape[1], image.shape[2]
                bounds = []
                for _ in range(self.pixel_count):
                    # x, y coordinates (pixel position)
                    bounds.extend([(0, h-1), (0, w-1)])
                    # RGB values for the pixel
                    bounds.extend([(0, 1), (0, 1), (0, 1)])

                # Perform differential evolution optimization to find optimal pixels
                result = differential_evolution(
                    self._evaluate_pixel_attack,
                    bounds,
                    args=(image, label),
                    maxiter=self.max_iter,
                    popsize=self.popsize,
                    mutation=(0.5, 1),
                    recombination=0.7
                )

                # Apply the optimized attack pixels to the image
                perturbed = self._apply_perturbation(image.clone(), result.x)
                perturbed_images[i] = perturbed
            
            return original_images, perturbed_images, labels
            
        except Exception as e:
            logging.error(f"Error in One Pixel attack: {str(e)}", exc_info=True)
            # Return original images if attack fails
            return images.detach(), images.detach(), labels.detach()
        finally:
            # Restore model's training mode and clean up memory
            if was_training:
                self.model.train()
            torch.cuda.empty_cache()
            gc.collect()

    def generate(self, images, labels, epsilon=None):
        """Generate adversarial examples with optional epsilon override"""
        try:
            # Save original parameters if we're overriding
            if epsilon is not None:
                original_epsilon = self.epsilon
                original_pixel_count = self.pixel_count
                self.epsilon = epsilon
                # Scale pixel count based on epsilon 
                self.pixel_count = max(1, int(self.pixel_count * (epsilon / original_epsilon)))
                
            # Call attack and return only the adversarial samples
            _, perturbed_images, _ = self.attack(images, labels)
            
            # Restore original parameters if they were overridden
            if epsilon is not None:
                self.epsilon = original_epsilon
                self.pixel_count = original_pixel_count
                
            return perturbed_images
            
        except Exception as e:
            logging.error(f"Error in One Pixel generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails

    def _evaluate_pixel_attack(self, perturbation, image, target_label):
        """Evaluate attack effectiveness"""
        try:
            perturbed = self._apply_perturbation(image.clone(), perturbation)
            output = self.model(perturbed.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            
            # Use negative probability of correct class as loss
            prob = torch.softmax(output, dim=1)[0, target_label].item()
            loss = prob  # We want to minimize this probability
            
            # If we've successfully changed the classification, return a low value
            if predicted.item() != target_label:
                return -1.0
                
            return loss
            
        except Exception as e:
            logging.error(f"Error in evaluation: {str(e)}")
            return 1.0  # Return high loss on error

    def _apply_perturbation(self, image, perturbation):
        """Apply pixel perturbation to the image"""
        perturbation = perturbation.reshape(-1, 5)  # [x, y, r, g, b]
        perturbed = image.clone()
        
        for p in perturbation:
            x, y = int(p[0]), int(p[1])
            if image.shape[0] == 3:  # RGB image
                perturbed[:, x, y] = torch.tensor([p[2], p[3], p[4]], device=self.device)
            else:  # Grayscale image
                perturbed[:, x, y] = torch.mean(torch.tensor([p[2], p[3], p[4]], device=self.device))
                
        return perturbed

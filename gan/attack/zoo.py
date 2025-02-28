import torch
import logging
import gc
import numpy as np


class ZooAttack:
    def __init__(self, model, epsilon=0.01, iterations=100, h=0.001, binary_search_steps=5):
        self.model = model
        self.epsilon = epsilon
        self.iterations = iterations
        self.h = h  # Step size for gradient estimation
        self.binary_search_steps = binary_search_steps
        self.device = next(model.parameters()).device
        self._logged = False
        logging.info("Zoo Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing Zoo (zeroth order optimization) attack.")
            self._logged = True
            
        # Store original training mode and set to eval
        was_training = self.model.training
        self.model.eval()
        
        try:
            images = images.to(self.device)
            labels = labels.to(self.device)
            # Clone original images to return later
            original_images = images.clone().detach()
            batch_size = images.shape[0]
            
            perturbed_images = images.clone().detach()
            
            # Attack each image individually
            for i in range(batch_size):
                img = images[i:i+1]
                label = labels[i:i+1]
                
                # Initial perturbation
                delta = torch.zeros_like(img, requires_grad=False).to(self.device)
                
                # Binary search for finding optimal c value
                c = self.epsilon
                c_lower = 0
                c_upper = 1.0
                
                for binary_step in range(self.binary_search_steps):
                    best_loss = float('inf')
                    best_delta = delta.clone()
                    
                    for _ in range(self.iterations):
                        # Select random pixel coordinate
                        x, y = np.random.randint(0, img.shape[2]), np.random.randint(0, img.shape[3])
                        
                        # For each color channel
                        for channel in range(img.shape[1]):
                            # Estimate gradient using finite difference method
                            img_plus_h = img.clone()
                            img_plus_h[0, channel, x, y] += self.h
                            img_minus_h = img.clone()
                            img_minus_h[0, channel, x, y] -= self.h
                            
                            # Compute loss for both directions
                            with torch.no_grad():
                                logits_plus = self.model(img_plus_h)
                                logits_minus = self.model(img_minus_h)
                                
                                # Target label is the original label
                                loss_plus = torch.nn.functional.cross_entropy(logits_plus, label)
                                loss_minus = torch.nn.functional.cross_entropy(logits_minus, label)
                                
                                # Approximate gradient
                                grad = (loss_plus - loss_minus) / (2 * self.h)
                                
                            # Update delta
                            delta[0, channel, x, y] -= c * torch.sign(grad)
                            
                            # Project perturbation to preserve constraints
                            perturbed = torch.clamp(img + delta, 0, 1)
                            delta = perturbed - img
                            
                            # Evaluate the perturbed image
                            with torch.no_grad():
                                logits = self.model(img + delta)
                                loss = torch.nn.functional.cross_entropy(logits, label)
                                
                                # Keep track of best result
                                if loss < best_loss:
                                    best_loss = loss
                                    best_delta = delta.clone()
                                    
                    # Update c with binary search
                    pred = self.model(img + best_delta).argmax(dim=1)
                    if pred != label:
                        c_upper = c
                        c = (c_lower + c_upper) / 2
                    else:
                        c_lower = c
                        c = (c_lower + c_upper) / 2
                        
                    delta = best_delta.clone()
                
                # Apply the final perturbation
                perturbed_images[i:i+1] = torch.clamp(img + delta, 0, 1)
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return original_images, perturbed_images, labels
            
        except Exception as e:
            logging.error(f"Error in Zoo attack: {str(e)}", exc_info=True)
            # Return original images if attack fails
            return images.detach(), images.detach(), labels.detach()
        finally:
            # Restore model's training mode
            if was_training:
                self.model.train()

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
            logging.error(f"Error in Zoo generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails

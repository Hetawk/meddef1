import torch
import logging
import gc
import numpy as np

class BoundaryAttack:
    def __init__(self, model, epsilon=0.3, steps=50, spherical_step=0.01, source_step=0.01, 
                 step_adaptation=1.5, max_directions=25):
        self.model = model
        self.epsilon = epsilon  # For API consistency
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.step_adaptation = step_adaptation
        self.max_directions = max_directions
        self.device = next(model.parameters()).device
        self._logged = False
        logging.info("Boundary Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing Boundary attack.")
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
            
            # Process each image individually
            for i in range(batch_size):
                img = images[i:i+1]
                label = labels[i:i+1]
                
                # Start with a random adversarial example as reference
                with torch.no_grad():
                    random_img = torch.rand_like(img).to(self.device)
                    output = self.model(random_img)
                    pred = output.argmax(dim=1)
                    
                    # Find a misclassified starting point
                    attempts = 0
                    while pred == label and attempts < 100:
                        random_img = torch.rand_like(img).to(self.device)
                        output = self.model(random_img)
                        pred = output.argmax(dim=1)
                        attempts += 1
                
                if pred == label:
                    # If we can't find a misclassified example, use an ineffective attack
                    logging.info("Could not find initial adversarial example.")
                    perturbed_images[i:i+1] = img
                    continue
                
                # Initialize variables for Boundary Attack
                current = random_img.clone()
                spherical_step = self.spherical_step
                source_step = self.source_step
                
                # Main attack loop
                for step in range(self.steps):
                    # Calculate perturbation vector
                    perturbation = current - img
                    perturbation_norm = torch.norm(perturbation)
                    
                    if perturbation_norm < self.epsilon:
                        # Already within epsilon ball, so we're done
                        break
                    
                    successful_directions = 0
                    
                    # Try multiple random directions
                    for _ in range(self.max_directions):
                        # Generate random direction
                        direction = torch.randn_like(img).to(self.device)
                        direction_norm = torch.norm(direction)
                        
                        if direction_norm < 1e-6:
                            continue
                            
                        direction = direction / direction_norm
                        
                        # Orthogonal step with spherical_step
                        orthogonal = current + spherical_step * perturbation_norm * direction
                        
                        # Move towards source image with source_step
                        next_candidate = orthogonal + source_step * (img - orthogonal)
                        
                        # Ensure the candidate is valid
                        next_candidate = torch.clamp(next_candidate, 0, 1)
                        
                        # Check if the candidate is still adversarial
                        with torch.no_grad():
                            output = self.model(next_candidate)
                            pred = output.argmax(dim=1)
                            
                        if pred != label:
                            # Success, update the current adversarial example
                            current = next_candidate.clone()
                            successful_directions += 1
                            break
                    
                    # Adjust step sizes
                    if successful_directions == 0:
                        spherical_step /= self.step_adaptation
                        source_step /= self.step_adaptation
                    else:
                        spherical_step *= self.step_adaptation
                        source_step *= self.step_adaptation
                
                perturbed_images[i:i+1] = current
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return original_images, perturbed_images, labels
            
        except Exception as e:
            logging.error(f"Error in Boundary attack: {str(e)}", exc_info=True)
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
            logging.error(f"Error in Boundary generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails
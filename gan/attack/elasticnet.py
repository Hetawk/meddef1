import torch
import logging
import gc

class ElasticNetAttack:
    def __init__(self, model, epsilon=0.3, alpha=0.01, iterations=40, beta=1.0):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.beta = beta
        self.device = next(model.parameters()).device
        self._logged = False
        logging.info("ElasticNet Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing ElasticNet attack.")
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
            perturbed_images.requires_grad = True
            loss_fn = torch.nn.CrossEntropyLoss()

            for _ in range(self.iterations):
                # Reset gradients
                if perturbed_images.grad is not None:
                    perturbed_images.grad.zero_()
                
                # Forward pass
                outputs = self.model(perturbed_images)
                cost = loss_fn(outputs, labels)
                
                # Backward pass
                cost.backward()
                grad = perturbed_images.grad
                
                # Apply ElasticNet perturbation
                perturbation = self.epsilon * torch.sign(grad) + self.alpha * grad
                perturbed_images = perturbed_images.detach() + self.beta * perturbation
                
                # Project perturbation to valid image range
                perturbed_images = torch.clamp(perturbed_images, 0, 1)
                perturbed_images.requires_grad = True

            # Clean up and return results
            perturbed_images = perturbed_images.detach()
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
            
            return original_images, perturbed_images, labels
            
        except Exception as e:
            logging.error(f"Error in ElasticNet attack: {str(e)}", exc_info=True)
            # Return original images if attack fails
            return images.detach(), images.detach(), labels.detach()
        finally:
            # Restore model's training mode
            if was_training:
                self.model.train()

    def generate(self, images, labels, epsilon=None):
        """Generate adversarial examples with optional epsilon override"""
        try:
            # Save original parameters if we're overriding
            if epsilon is not None:
                original_epsilon = self.epsilon
                self.epsilon = epsilon
                
            # Call attack and return only the adversarial samples
            _, perturbed_images, _ = self.attack(images, labels)
            
            # Restore original parameters if they were overridden
            if epsilon is not None:
                self.epsilon = original_epsilon
                
            return perturbed_images
            
        except Exception as e:
            logging.error(f"Error in ElasticNet generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails

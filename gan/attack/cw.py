import torch
import torch.nn as nn
import logging
import gc

class CWAttack:
    def __init__(self, model, epsilon=0.3, c=1.0, iterations=100, lr=0.01, binary_search_steps=9, confidence=0):
        self.model = model
        self.epsilon = epsilon  # For API consistency, but CW uses c instead
        self.c = c  # Coefficient for objective function
        self.iterations = iterations
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.confidence = confidence
        self.device = next(model.parameters()).device
        self._logged = False
        logging.info("CW L2 Attack initialized.")

    def attack(self, images, labels):
        if not self._logged:
            logging.info("Performing CW attack.")
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
            
            # Attack each image individually (CW attack can be complex for batches)
            for i in range(batch_size):
                img = images[i:i+1]
                target = labels[i:i+1]
                
                # Use binary search to find the optimal c value
                c_lower = 0
                c_upper = 10.0
                c = self.c
                
                best_adv = None
                best_l2 = float('inf')
                
                for _ in range(self.binary_search_steps):
                    # Initialize optimization variable (w)
                    w = torch.zeros_like(img).to(self.device)
                    w.requires_grad = True
                    optimizer = torch.optim.Adam([w], lr=self.lr)
                    
                    for step in range(self.iterations):
                        # Convert w to adversarial example
                        adv = torch.tanh(w) * 0.5 + 0.5  # Map to [0,1]
                        
                        # Calculate L2 distance
                        l2_dist = torch.norm(adv - img, p=2)
                        
                        # Calculate CW loss
                        outputs = self.model(adv)
                        correct_logit = outputs.gather(1, target.unsqueeze(1)).squeeze()
                        wrong_logit = outputs.clone()
                        wrong_logit[0, target] = -float('inf')
                        wrong_logit = wrong_logit.max(1)[0]
                        
                        # f(x') = max(Z(x')_t - max{Z(x')_i : i ≠ t}, -confidence)
                        f_loss = torch.clamp(correct_logit - wrong_logit + self.confidence, min=0)
                        
                        # Total loss: distance + c * f(x')
                        loss = l2_dist + c * f_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Check if this is a better solution
                        with torch.no_grad():
                            adv_out = self.model(adv)
                            adv_pred = adv_out.argmax(dim=1)
                            
                            if adv_pred != target and l2_dist < best_l2:
                                best_l2 = l2_dist
                                best_adv = adv.clone()
                    
                    # Update c with binary search
                    if best_adv is None:
                        c_lower = c
                        c = (c_lower + c_upper) / 2
                    else:
                        c_upper = c
                        c = (c_lower + c_upper) / 2
                
                # Apply the best adversarial example found
                if best_adv is not None:
                    perturbed_images[i:i+1] = best_adv
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return original_images, perturbed_images, labels
            
        except Exception as e:
            logging.error(f"Error in CW attack: {str(e)}", exc_info=True)
            # Return original images if attack fails
            return images.detach(), images.detach(), labels.detach()
        finally:
            # Restore model's training mode
            if was_training:
                self.model.train()

    def generate(self, images, labels, epsilon=None):
        """Generate adversarial examples with optional epsilon override"""
        try:
            # Save original params if we're overriding
            if epsilon is not None:
                original_epsilon = self.epsilon
                original_c = self.c
                self.epsilon = epsilon
                self.c = epsilon * 10  # Scale c based on epsilon
                
            # Call attack and return only the adversarial samples
            _, perturbed_images, _ = self.attack(images, labels)
            
            # Restore original params if they were overridden
            if epsilon is not None:
                self.epsilon = original_epsilon
                self.c = original_c
                
            return perturbed_images
            
        except Exception as e:
            logging.error(f"Error in CW generate: {str(e)}", exc_info=True)
            return images  # Return original images if attack fails

# adv_train.py

import torch
from gan.attack.attack_loader import AttackLoader
from gan.attack.attack_data_loader import AttackDataLoader
import logging
import os
from torchvision.utils import save_image
import gc
from utils.robustness.training_config import AdversarialTrainingConfig
import math


class AdversarialTraining:
    """Adversarial Training Implementation"""

    def __init__(self, model, criterion, config):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = next(model.parameters()).device

        # Get attack parameters from config
        self.attack_type = getattr(config, 'attack_type', ['fgsm'])[0]
        self.epsilon = getattr(config, 'attack_eps', 0.1)
        self.attack_steps = getattr(config, 'attack_steps', 10)
        self.attack_alpha = getattr(config, 'attack_alpha', 0.01)

        # Get adversarial training parameters
        if hasattr(config, 'initial_epsilon') and config.initial_epsilon is not None:
            self.initial_epsilon = config.initial_epsilon
            # Handle case where initial_epsilon > epsilon (user wants aggressive start)
            if self.initial_epsilon > self.epsilon:
                logging.warning(
                    f"Initial epsilon ({self.initial_epsilon}) is larger than target epsilon ({self.epsilon}).")
                logging.warning(
                    "This is an aggressive adversarial training strategy.")
                # We'll keep the value as requested but cap final epsilon to match target
                self.final_epsilon = self.epsilon
                self.epsilon_increasing = False  # We'll be decreasing epsilon instead
            else:
                self.final_epsilon = self.epsilon
                self.epsilon_increasing = True
        else:
            # Default to using a variable-phase training approach
            # Start with low epsilon, increase gradually
            self.initial_epsilon = 0.01  # Start very small - safety first
            self.final_epsilon = self.epsilon  # Target value
            self.epsilon_increasing = True
            self.adaptive_schedule = True

        self.epsilon_steps = getattr(config, 'epsilon_steps', 5)
        self.initial_adv_weight = getattr(config, 'initial_adv_weight', 0.2)
        self.adv_weight = getattr(config, 'adv_weight', 0.5)
        self.dynamic_alpha = getattr(config, 'dynamic_alpha', True)
        self.epsilon_schedule = getattr(config, 'epsilon_schedule', 'linear')

        # Create the attack
        self.attack_loader = AttackLoader(model, config)
        self.attack = self.attack_loader.get_attack(self.attack_type)

        # Set initial values
        self.current_epsilon = self.initial_epsilon  # Start with initial epsilon
        self.clean_weight = 1.0 - self.initial_adv_weight
        self.adv_weight = self.initial_adv_weight
        self.alpha = self.attack_alpha

        # Initialize attack data cache
        self.attack_cache = {}

        # Safety thresholds to prevent catastrophic collapse
        self.max_safe_epsilon = 0.03  # More conservative value
        self.epsilon_growth_factor = 1.0

        # Remove ANFIS-related code
        self.use_anfis = False

        # Log initialization
        logging.info("Adversarial training initialized with:")
        logging.info(f" - Initial epsilon: {self.initial_epsilon}")
        logging.info(f" - Final epsilon: {self.final_epsilon}")
        logging.info(
            f" - Epsilon schedule: {self.epsilon_schedule} over {self.epsilon_steps} epochs")
        logging.info(f" - Initial adv weight: {self.initial_adv_weight}")
        logging.info(f" - Final adv weight: {self.adv_weight}")
        logging.info(f" - {self.attack_type} steps: {self.attack_steps}")
        if self.dynamic_alpha:
            logging.info(" - Using dynamic PGD step size (alpha)")

        # Enhanced dynamic parameters for stability
        self.adaptive_schedule = getattr(config, 'adaptive_schedule', True)
        self.warmup_epochs = 8  # Increased for stability
        self.aggressive_epochs = 15
        self.stabilization_epochs = 8

        # Conservative parameters for medical imaging
        self.max_safe_epsilon = 0.03  # Lower maximum epsilon
        self.initial_max_epsilon = 0.02  # Lower initial cap during warmup
        self.stabilization_epsilon = 0.015  # Safer fallback epsilon

        # NaN recovery
        self.nan_counter = 0
        self.max_nan_tolerance = 3
        self.global_nan_count = 0  # Track NaNs across entire training
        self.last_stable_epsilon = self.initial_epsilon
        self.max_global_nan = 5
        self.nan_detected_recently = False

        # Auto-recovery mechanisms
        self.recovery_triggered = False
        self.stability_threshold = 0.5
        self.consecutive_drops = 0
        self.max_drops = 3

    def update_parameters(self, epoch):
        """Update epsilon and adversarial weight based on current epoch with enhanced scheduling"""
        # Calculate progress through schedule (0 to 1)
        progress = min(
            1.0, epoch / self.epsilon_steps) if self.epsilon_steps > 0 else 1.0

        # Store the last epsilon value for safety
        self.last_stable_epsilon = self.current_epsilon

        # If using adaptive schedule (the enhanced approach)
        if getattr(self, 'adaptive_schedule', False):
            total_phases = self.warmup_epochs + \
                self.aggressive_epochs + self.stabilization_epochs

            # Check if we've had too many NaNs globally and throttle accordingly
            if self.global_nan_count > self.max_nan_tolerance:
                # Stop increasing epsilon and use conservative values
                self.current_epsilon = min(0.01, self.last_stable_epsilon)
                logging.warning(
                    f"Multiple NaNs detected across training. Using conservative epsilon={self.current_epsilon:.4f}")

                # Also reduce adv_weight when unstable
                self.adv_weight = 0.2
                self.clean_weight = 0.8
                return

            if epoch < self.warmup_epochs:
                # Warmup phase - more conservative for medical images
                # Never exceed initial_max_epsilon during warmup
                phase_progress = epoch / self.warmup_epochs
                target = min(0.5 * self.final_epsilon,
                             self.initial_max_epsilon)
                self.current_epsilon = self.initial_epsilon + \
                    phase_progress * (target - self.initial_epsilon)
                logging.info(
                    f"Epoch {epoch+1}: Warmup phase - epsilon={self.current_epsilon:.4f}")

            elif epoch < (self.warmup_epochs + self.aggressive_epochs):
                # Aggressive phase - modified for smoother transition
                phase_progress = (epoch - self.warmup_epochs) / \
                    self.aggressive_epochs

                # Use a quadratic increase for smoother transitions
                quad_progress = phase_progress ** 1.5  # Slower growth
                self.current_epsilon = min(
                    self.initial_max_epsilon + quad_progress *
                    (self.final_epsilon - self.initial_max_epsilon),
                    self.max_safe_epsilon
                )

                logging.info(
                    f"Epoch {epoch+1}: Aggressive phase - epsilon={self.current_epsilon:.4f}")

            else:
                # Stabilization phase - hold steady at final epsilon
                self.current_epsilon = min(
                    self.final_epsilon, self.max_safe_epsilon)
                logging.info(
                    f"Epoch {epoch+1}: Stabilization phase - epsilon={self.current_epsilon:.4f}")

            # Update adv_weight to be higher during aggressive phase
            if epoch < self.warmup_epochs:
                # Warmup phase - lower weight on adversarial examples
                self.adv_weight = self.initial_adv_weight
            elif epoch < (self.warmup_epochs + self.aggressive_epochs):
                # Aggressive phase - gradually increase adv_weight to final value
                phase_progress = (epoch - self.warmup_epochs) / \
                    self.aggressive_epochs
                self.adv_weight = self.initial_adv_weight + phase_progress * \
                    (self.adv_weight - self.initial_adv_weight)

            # Clean weight is complement of adv_weight
            self.clean_weight = 1.0 - self.adv_weight

        else:
            # Standard non-adaptive approach
            if self.epsilon_increasing:
                # Regular case: start low, increase to target epsilon
                if self.epsilon_schedule == 'linear':
                    # Linear schedule
                    self.current_epsilon = self.initial_epsilon + \
                        progress * (self.final_epsilon - self.initial_epsilon)
                elif self.epsilon_schedule == 'exponential':
                    # Exponential schedule (slower start, faster finish)
                    self.current_epsilon = self.initial_epsilon + \
                        (1 - math.exp(-5 * progress)) * \
                        (self.final_epsilon - self.initial_epsilon)
                elif self.epsilon_schedule == 'cosine':
                    # Cosine schedule (faster start, slower finish)
                    self.current_epsilon = self.initial_epsilon + 0.5 * \
                        (1 - math.cos(math.pi * progress)) * \
                        (self.final_epsilon - self.initial_epsilon)
            else:
                # Special case: start high, decrease to target epsilon
                if self.epsilon_schedule == 'linear':
                    # Linear decrease
                    self.current_epsilon = self.initial_epsilon - \
                        progress * (self.initial_epsilon - self.final_epsilon)
                elif self.epsilon_schedule == 'exponential':
                    # Exponential decrease (slower start, faster finish)
                    self.current_epsilon = self.initial_epsilon - \
                        (1 - math.exp(-5 * progress)) * \
                        (self.initial_epsilon - self.final_epsilon)
                elif self.epsilon_schedule == 'cosine':
                    # Cosine decrease (faster start, slower finish)
                    self.current_epsilon = self.initial_epsilon - 0.5 * \
                        (1 - math.cos(math.pi * progress)) * \
                        (self.initial_epsilon - self.final_epsilon)

        # Apply safety cap to epsilon to prevent training instability
        self.current_epsilon = min(
            self.current_epsilon, self.max_safe_epsilon)

        # Reset epsilon to a safer value if NaN was recently detected
        if getattr(self, 'nan_detected_recently', False):
            # Go back to a safer epsilon (60% of current planned value)
            self.current_epsilon = min(
                self.current_epsilon * 0.5, self.last_stable_epsilon)
            logging.warning(
                f"Using safer epsilon value after NaN: {self.current_epsilon:.4f}")
            self.nan_detected_recently = False

        # Also update alpha if using dynamic alpha - with safer values
        if self.dynamic_alpha:
            # More conservative alpha values, especially at higher epsilons
            self.alpha = min(self.current_epsilon / 6.0, 0.003)

        # Ensure weights sum to 1.0
        weight_sum = self.clean_weight + self.adv_weight
        if weight_sum != 1.0:
            self.clean_weight = self.clean_weight / weight_sum
            self.adv_weight = self.adv_weight / weight_sum

    def reduce_epsilon_growth(self, factor=0.8):
        """Reduce the epsilon growth factor when training becomes unstable"""
        self.epsilon_growth_factor *= factor
        logging.warning(
            f"Reduced epsilon growth factor to {self.epsilon_growth_factor:.2f}")

        # Also reduce current epsilon immediately
        original_epsilon = self.current_epsilon
        self.current_epsilon = max(
            self.initial_epsilon, self.current_epsilon * factor)
        logging.warning(
            f"Reduced epsilon from {original_epsilon:.4f} to {self.current_epsilon:.4f}")

    def adversarial_loss(self, data, target, batch_indices=None):
        try:
            # Update attack parameters
            if hasattr(self.attack, 'epsilon'):
                self.attack.eps = self.current_epsilon
            if hasattr(self.attack, 'alpha'):
                self.attack.alpha = self.alpha

            # Create a clean copy that requires gradients
            # Use clone to ensure we're not modifying the original tensor
            x = data.clone().detach()
            x.requires_grad_(True)

            # FIX: We'll use our own PGD implementation instead of relying on the attack object
            # This gives us more control over memory and stability
            if self.attack_type.lower() == 'pgd':
                adv_data = self._pgd_attack(x, target)
            else:
                # For non-PGD attacks, use the attack object with safety checks
                try:
                    if hasattr(self.attack, 'generate'):
                        adv_data = self.attack.generate(
                            x, target, self.current_epsilon)
                    else:
                        _, adv_data, _ = self.attack.attack(x, target)
                except Exception as e:
                    logging.warning(
                        f"Error generating adversarial examples: {e}")
                    # Fall back to a simple FGSM-like perturbation for stability
                    adv_data = self._fgsm_attack(x, target)

            # Check for NaN values in adversarial examples
            if torch.isnan(adv_data).any() or torch.isinf(adv_data).any():
                logging.warning(
                    "NaN/Inf detected in adversarial examples - using clean data + small noise")
                adv_data = x.detach() + torch.randn_like(x) * 0.001
                self.nan_detected_recently = True
                self.global_nan_count += 1

            # Store original data for loss calculation
            orig_data = data.clone().detach()

            # Forward pass through model with adversarial examples
            # Make sure to detach adv_data to prevent gradient flow issues
            adv_data = adv_data.detach()

            try:
                adv_output = self.model(adv_data)

                # Check for NaN in outputs
                if torch.isnan(adv_output).any() or torch.isinf(adv_output).any():
                    logging.warning("NaN/Inf detected in adversarial outputs")

                    # Reset batch norm statistics if present
                    for m in self.model.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.reset_running_stats()

                    # Fall back to clean data
                    adv_output = self.model(orig_data)
                    self.nan_detected_recently = True

                # Calculate loss with NaN protection
                try:
                    adv_loss = self.criterion(adv_output, target)
                    if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                        raise ValueError("NaN/Inf in adversarial loss")
                except Exception:
                    # Use a stable but high loss value
                    adv_loss = torch.tensor(2.0, device=data.device)
                    self.nan_detected_recently = True
                    logging.warning(
                        "NaN in adversarial loss - using fixed loss value")

            except Exception as e:
                logging.warning(f"Error in adversarial forward pass: {e}")
                # Graceful fallback - use clean data
                adv_output = self.model(orig_data)
                adv_loss = self.criterion(adv_output, target)
                self.nan_detected_recently = True

            # Save samples only once
            if not getattr(self, '_attack_samples_saved', False):
                self.save_attack_samples(orig_data, adv_data)
                self._attack_samples_saved = True

            return adv_loss

        except Exception as e:
            logging.exception(
                "Error occurred during adversarial loss calculation:")
            # Return small non-zero loss
            self.nan_detected_recently = True
            return torch.tensor(1.0, device=data.device)

    def _pgd_attack(self, x, target, random_start=True):
        """
        Custom PGD implementation with improved memory efficiency and stability.
        This avoids in-place operations that break autograd.
        """
        # Clone and detach input for safety
        x_adv = x.clone().detach()

        # Random start initialization
        if random_start:
            # Create random noise within epsilon bounds
            noise = torch.zeros_like(
                x_adv).uniform_(-self.current_epsilon, self.current_epsilon)
            # Add noise and clip to valid image range
            x_adv = torch.clamp(x_adv + noise, 0, 1)

        for _ in range(self.attack_steps):
            # Ensure requires_grad is set
            x_adv.requires_grad_(True)

            # Forward pass and loss calculation
            with torch.enable_grad():
                outputs = self.model(x_adv)
                loss = self.criterion(outputs, target)

            # Get gradients
            grad = torch.autograd.grad(loss, x_adv,
                                       retain_graph=False,
                                       create_graph=False)[0]

            # Detach x_adv from computation graph before updating
            x_adv = x_adv.detach()

            # Create adversarial example - avoid in-place operations
            with torch.no_grad():
                # Use sign of gradient for stability
                grad_sign = torch.sign(grad.detach())

                # Take a step in the gradient direction
                x_adv = x_adv + self.alpha * grad_sign

                # Project back to epsilon ball around original x
                delta = torch.clamp(
                    x_adv - x, -self.current_epsilon, self.current_epsilon)
                x_adv = x + delta

                # Ensure valid image range
                x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

    def _fgsm_attack(self, x, target):
        """Simple FGSM attack as a fallback for stability"""
        x_adv = x.clone()
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = self.model(x_adv)
        loss = self.criterion(outputs, target)

        # Get gradients
        loss.backward()

        # Create perturbation (avoid in-place operations)
        with torch.no_grad():
            grad_sign = torch.sign(x_adv.grad)
            perturbation = self.current_epsilon * grad_sign
            x_adv = x + perturbation
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.detach()

    def generate_attacks_in_batches(self, loader, split='train', max_samples=None):
        """Generate adversarial examples in batches to manage memory"""
        self.model.eval()
        all_originals = []
        all_adversarials = []
        all_labels = []
        total_processed = 0

        try:
            with torch.enable_grad():
                for batch_idx, (data, target) in enumerate(loader):
                    if max_samples and total_processed >= max_samples:
                        break

                    # Move batch to device
                    data = data.to(self.device)
                    target = target.to(self.device)

                    # Generate adversarial examples using our stable implementation
                    if self.attack_type.lower() == 'pgd':
                        adv_data = self._pgd_attack(data, target)
                    else:
                        # Use the attack API
                        try:
                            if hasattr(self.attack, 'generate'):
                                adv_data = self.attack.generate(
                                    data, target, self.current_epsilon)
                            else:
                                _, adv_data, _ = self.attack.attack(
                                    data, target)
                        except Exception:
                            # Fallback to FGSM
                            adv_data = self._fgsm_attack(data, target)

                    # Move results to CPU
                    all_originals.append(data.cpu())
                    all_adversarials.append(adv_data.cpu())
                    all_labels.append(target.cpu())

                    total_processed += len(data)

                    # Clear GPU memory
                    del adv_data
                    torch.cuda.empty_cache()

                    if batch_idx % 10 == 0:
                        logging.info(f"Processed {total_processed} samples...")

                    # Save intermediate results if memory usage is high
                    if hasattr(self, 'batch_size') and hasattr(self, 'max_samples_in_memory'):
                        if len(all_originals) * self.batch_size >= self.max_samples_in_memory:
                            self._save_intermediate_results(
                                all_originals, all_adversarials, all_labels,
                                split, total_processed)
                            all_originals = []
                            all_adversarials = []
                            all_labels = []
                            gc.collect()

        except RuntimeError as e:
            logging.error(f"Runtime error during attack generation: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error generating attacks: {str(e)}")
            return None
        finally:
            self.model.train()

        return all_originals, all_adversarials, all_labels

    def _save_intermediate_results(self, originals, adversarials, labels, split, current_count):
        """Save intermediate results to disk"""
        try:
            output_dir = os.path.join(
                'out', 'attacks', split, f'batch_{current_count}')
            os.makedirs(output_dir, exist_ok=True)

            # Concatenate batches
            orig = torch.cat(originals, dim=0)
            adv = torch.cat(adversarials, dim=0)
            labs = torch.cat(labels, dim=0)

            # Save to disk
            torch.save({
                'original': orig,
                'adversarial': adv,
                'labels': labs
            }, os.path.join(output_dir, 'attacks.pt'))

            logging.info(
                f"Saved intermediate results for {current_count} samples")
        except Exception as e:
            logging.error(f"Error saving intermediate results: {str(e)}")

    def save_attack_samples(self, orig, adv_data):
        """Save samples of adversarial examples for visualization"""
        try:
            # Determine folder structure
            task = getattr(self.config, 'task_name', 'default_task')

            # Handle dataset name
            if hasattr(self.config, 'data_key'):
                dataset = self.config.data_key
            elif hasattr(self.config, 'data'):
                if isinstance(self.config.data, list):
                    dataset = self.config.data[0]
                else:
                    dataset = self.config.data
            else:
                dataset = 'default_dataset'

            # Handle model name
            if hasattr(self.config, 'model_name'):
                model_name = self.config.model_name
            elif hasattr(self.config, 'arch') and hasattr(self.config, 'depth'):
                arch = self.config.arch[0] if isinstance(
                    self.config.arch, list) else self.config.arch
                depth_val = None
                if isinstance(self.config.depth, dict):
                    depth_list = self.config.depth.get(arch, [])
                    if depth_list:
                        depth_val = depth_list[0]
                else:
                    depth_val = self.config.depth
                model_name = f"{arch}_{depth_val}" if depth_val is not None else self.model.__class__.__name__
            else:
                model_name = self.model.__class__.__name__

            # Handle attack name
            if isinstance(getattr(self.config, 'attack_name', None), list):
                attack = "+".join(self.config.attack_name)
            else:
                attack = getattr(self.config, 'attack_name', self.attack_type)

            folder = os.path.join("out", task, dataset,
                                  model_name, "attack", attack)
            os.makedirs(folder, exist_ok=True)

            # Save samples
            num_samples = min(5, adv_data.size(0))
            for i in range(num_samples):
                # Create filenames
                orig_filename = os.path.join(folder, f"sample_{i}_orig.png")
                adv_filename = os.path.join(folder, f"sample_{i}_adv.png")
                pert_filename = os.path.join(
                    folder, f"sample_{i}_perturbation.png")

                # Save original and adversarial images
                save_image(orig[i], orig_filename)
                save_image(adv_data[i], adv_filename)

                # Calculate perturbation safely
                with torch.no_grad():
                    perturbation = adv_data[i].clone() - orig[i].clone()

                    # Save raw perturbation
                    save_image(perturbation, pert_filename)

                    # Enhanced perturbation visualization
                    enhanced_pert = (perturbation * 5) + 0.5
                    save_image(enhanced_pert, pert_filename)

                    # Extreme enhancement for subtle perturbations
                    extreme_pert = (perturbation * 20) + 0.5
                    extreme_pert_filename = os.path.join(
                        folder, f"sample_{i}_perturbation_enhanced.png")
                    save_image(extreme_pert, extreme_pert_filename)

            # Calculate perturbation statistics
            with torch.no_grad():
                perturbation_tensor = adv_data[:num_samples] - \
                    orig[:num_samples]
                perturbations = perturbation_tensor.view(num_samples, -1)
                avg_norm = torch.norm(perturbations, p=2, dim=1).mean().item()

            # Write summary
            summary_path = os.path.join(folder, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("Attack Summary\n")
                f.write("======================\n")
                f.write(f"Attack Type: {attack}\n")
                f.write(f"Epsilon: {self.current_epsilon}\n")
                if hasattr(self.config, 'attack_alpha'):
                    f.write(f"Attack Alpha: {self.alpha}\n")
                if hasattr(self.config, 'attack_steps'):
                    f.write(f"Attack Steps: {self.attack_steps}\n")
                f.write(f"Number of samples saved: {num_samples}\n")
                f.write(f"Average Perturbation ℓ₂ Norm: {avg_norm:.4f}\n")

            logging.info(
                f"Saved {num_samples} adversarial samples and summary to {folder}")

            # Create enhanced visualization
            self._create_perturbation_visualization(
                orig[:num_samples], adv_data[:num_samples],
                os.path.join(folder, "perturbation_visualization.png"))

        except Exception as e:
            logging.exception("Exception in save_attack_samples:")

    def _create_perturbation_visualization(self, original, adversarial, save_path):
        """Create an enhanced visualization of perturbations"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Calculate perturbation safely
            with torch.no_grad():
                perturbation = adversarial.clone() - original.clone()

            # Convert to numpy for processing
            original_np = original.detach().cpu().numpy()
            adversarial_np = adversarial.detach().cpu().numpy()
            perturbation_np = perturbation.detach().cpu().numpy()

            # Scale perturbations for better visualization
            enhanced_pert1 = np.clip(perturbation_np * 5 + 0.5, 0, 1)
            enhanced_pert2 = np.clip(perturbation_np * 10 + 0.5, 0, 1)
            enhanced_pert3 = np.clip(perturbation_np * 20 + 0.5, 0, 1)

            # Create heatmap of perturbation magnitude
            heatmap = np.sqrt(
                np.sum(perturbation_np**2, axis=1, keepdims=True))
            heatmap = heatmap / (heatmap.max() + 1e-8)

            # Create figure
            num_samples = min(3, original.size(0))
            fig, axs = plt.subplots(
                num_samples, 6, figsize=(20, 4 * num_samples))

            # If only one sample, make sure axs is 2D
            if num_samples == 1:
                axs = axs.reshape(1, -1)

            # Column titles
            titles = ['Original', 'Adversarial', 'Perturbation (5x)',
                      'Perturbation (10x)', 'Perturbation (20x)', 'Heatmap']

            for i in range(num_samples):
                for j, (img, title) in enumerate(zip(
                        [original_np[i], adversarial_np[i], enhanced_pert1[i],
                         enhanced_pert2[i], enhanced_pert3[i], np.repeat(heatmap[i], 3, axis=0)],
                        titles)):

                    # Move channel to the end for matplotlib
                    img = np.transpose(img, (1, 2, 0))

                    # Handle grayscale images
                    if img.shape[2] == 1:
                        img = img.squeeze(2)

                    # Display the image
                    axs[i, j].imshow(img, cmap='viridis' if j == 5 else None)
                    axs[i, j].set_title(title)
                    axs[i, j].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            logging.info(
                f"Saved enhanced perturbation visualization to {save_path}")
        except Exception as e:
            logging.warning(f"Error creating perturbation visualization: {e}")

    def handle_nan_detected(self):
        """Handle the case when NaN is detected during training"""
        self.nan_counter += 1
        self.global_nan_count += 1
        self.nan_detected_recently = True

        # More aggressive reduction of epsilon
        original_epsilon = self.current_epsilon
        self.current_epsilon = min(
            self.stabilization_epsilon,  # Use known stable epsilon
            # Cut by 50% for safety
            max(self.initial_epsilon, self.last_stable_epsilon * 0.5)
        )

        logging.warning(f"NaN detected ({self.nan_counter}/{self.max_nan_tolerance}). "
                        f"Reducing epsilon from {original_epsilon:.4f} to {self.current_epsilon:.4f}")

        # If we've had too many NaNs, switch to safer training
        if self.nan_counter >= self.max_nan_tolerance:
            self.adaptive_schedule = False  # Disable adaptive schedule
            self.current_epsilon = self.initial_epsilon  # Reset to initial epsilon

            # Reduce adversarial weight
            self.adv_weight = max(0.1, self.adv_weight * 0.5)
            self.clean_weight = 1.0 - self.adv_weight

            logging.warning(
                f"Too many NaN events. Switching to conservative training with "
                f"epsilon={self.current_epsilon:.4f}, adv_weight={self.adv_weight:.2f}")

        return self.nan_counter >= self.max_nan_tolerance

    def react_to_metrics(self, val_accuracy, adv_val_accuracy=None):
        """React to validation metrics to ensure stability"""
        # Reset nan_detected flag when we have good validation results
        if val_accuracy > 0.75:
            self.nan_detected_recently = False
            self.last_stable_epsilon = self.current_epsilon

        # If validation accuracy drops too low, reduce epsilon to prevent collapse
        if val_accuracy < self.stability_threshold:
            if not self.recovery_triggered:
                self.recovery_triggered = True
                # Reduce epsilon by 30%
                self.current_epsilon *= 0.7
                logging.warning(
                    f"Training instability detected. Reducing epsilon to {self.current_epsilon:.4f}")
                self.consecutive_drops += 1

                # If multiple drops, reduce adv_weight too
                if self.consecutive_drops >= self.max_drops:
                    self.adv_weight *= 0.8
                    self.clean_weight = 1.0 - self.adv_weight
                    logging.warning(
                        f"Multiple instabilities detected. Reducing adv_weight to {self.adv_weight:.4f}")
        else:
            # Reset recovery flag when stable again
            self.recovery_triggered = False

# adv_train.py

import torch
from gan.attack.attack_loader import AttackLoader
from gan.attack.attack_data_loader import AttackDataLoader
import logging
import os
from torchvision.utils import save_image
import gc
from utils.robustness.training_config import AdversarialTrainingConfig


class AdversarialTraining:
    def __init__(self, model, criterion, config):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = next(model.parameters()).device

        # Ensure attack_name and epsilon are set from config
        if not hasattr(config, 'attack_name'):
            config.attack_name = getattr(config, 'attack_type', 'fgsm')
        if not hasattr(config, 'epsilon'):
            config.epsilon = getattr(config, 'attack_eps', 0.3)

        # Initialize epsilon scheduling parameters
        self.initial_epsilon = getattr(
            config, 'initial_epsilon', config.epsilon / 3)
        self.epsilon = self.initial_epsilon  # Start with smaller epsilon
        self.final_epsilon = config.epsilon
        self.epsilon_steps = getattr(config, 'epsilon_steps', 5)
        self.epsilon_schedule = getattr(
            config, 'epsilon_schedule', 'cosine')  # New parameter
        self.current_step = 0

        # Initialize adv weight scheduling parameters
        self.initial_adv_weight = getattr(config, 'initial_adv_weight', 0.2)
        self.adv_weight = self.initial_adv_weight
        self.final_adv_weight = getattr(config, 'adv_weight', 0.5)

        # Track clean and adv weights separately for loss calculation
        self.clean_weight = 1.0 - self.adv_weight

        # PGD attack parameters
        self.attack_steps = getattr(config, 'attack_steps', 7)
        self.attack_alpha = getattr(config, 'attack_alpha', 0.01)
        self.dynamic_alpha = getattr(
            config, 'dynamic_alpha', True)  # New parameter

        # Initialize the attack using AttackLoader
        self.attack_loader = AttackLoader(model, config)
        self.attack = self.attack_loader.get_attack(config.attack_name)

        # Log the attack type(s) being used
        if isinstance(config.attack_name, list):
            logging.info(
                f"Setting up adversarial training with attacks: {', '.join(config.attack_name)}")
        else:
            logging.info(
                f"Setting up adversarial training with attack: {config.attack_name}")

        if self.attack is None:
            raise ValueError(
                f"Failed to initialize attack {config.attack_name} for adversarial training")
        # Flag to ensure samples are saved only once
        self._attack_samples_saved = False

        # Initialize attack data loader
        self.attack_data = AttackDataLoader(
            dataset_name=config.data[0] if isinstance(
                config.data, list) else config.data,
            model_name=f"{config.arch[0]}_{config.depth[config.arch[0]][0]}" if isinstance(
                config.arch, list) else f"{config.arch}_{config.depth[config.arch][0]}",
            attack_type=config.attack_name
        )

        # Check if pre-generated attacks exist
        self.use_pregenerated = self.attack_data.validate_attacks_exist()
        if self.use_pregenerated:
            logging.info("Using pre-generated attacks for training")
        else:
            logging.warning(
                "No pre-generated attacks found, will generate attacks on-the-fly")

        self.batch_size = getattr(config, 'train_batch', 32)
        self.max_samples_in_memory = 1000  # Adjust based on available GPU memory

        # Print key parameters
        logging.info(f"Adversarial training initialized with:")
        logging.info(f" - Initial epsilon: {self.initial_epsilon}")
        logging.info(f" - Final epsilon: {self.final_epsilon}")
        logging.info(
            f" - Epsilon schedule: {self.epsilon_schedule} over {self.epsilon_steps} epochs")
        logging.info(f" - Initial adv weight: {self.initial_adv_weight}")
        logging.info(f" - Final adv weight: {self.final_adv_weight}")
        logging.info(f" - PGD steps: {self.attack_steps}")
        if self.dynamic_alpha:
            logging.info(f" - Using dynamic PGD step size (alpha)")
        else:
            logging.info(f" - Fixed PGD step size: {self.attack_alpha}")

    def update_parameters(self, epoch):
        """Update epsilon and adv_weight according to schedule"""
        # Use training config helper for more sophisticated scheduling
        self.epsilon = AdversarialTrainingConfig.get_epsilon_schedule(
            self.initial_epsilon,
            self.final_epsilon,
            self.epsilon_steps,
            epoch,
            self.epsilon_schedule
        )

        # Dynamic alpha calculation based on current epsilon
        if self.dynamic_alpha:
            self.attack_alpha = AdversarialTrainingConfig.get_pgd_alpha(
                self.epsilon,
                self.attack_steps
            )

        # Get loss weights that prioritize clean accuracy early, then robustness
        self.clean_weight, self.adv_weight = AdversarialTrainingConfig.get_combined_loss_weights(
            epoch,
            warmup_epochs=min(5, self.epsilon_steps // 2),
            final_adv_weight=self.final_adv_weight
        )

        logging.info(
            f"Epoch {epoch+1}: Updated adversarial parameters - "
            f"epsilon={self.epsilon:.4f}, "
            f"alpha={self.attack_alpha:.4f}, "
            f"clean_weight={self.clean_weight:.2f}, "
            f"adv_weight={self.adv_weight:.2f}"
        )

    def adversarial_loss(self, data, target, batch_indices=None):
        try:
            # Update attack parameters
            if hasattr(self.attack, 'epsilon'):
                self.attack.eps = self.epsilon
            if hasattr(self.attack, 'alpha'):
                self.attack.alpha = self.attack_alpha

            # Create tensor that explicitly requires gradients for attack
            # This is critical - we need to ensure gradients are enabled
            x = data.clone()
            x.requires_grad_(True)

            # For PGD we need to ensure no_grad is not active
            with torch.enable_grad():
                # Generate adversarial examples
                if hasattr(self.attack, 'generate'):
                    adv_data = self.attack.generate(x, target, self.epsilon)
                else:
                    _, adv_data, _ = self.attack.attack(x, target)

            # Forward pass through model with adversarial examples
            adv_output = self.model(adv_data)
            adv_loss = self.criterion(adv_output, target)

            # Save samples only once
            if not self._attack_samples_saved:
                self.save_attack_samples(x, adv_data)
                self._attack_samples_saved = True

            return adv_loss
        except Exception as e:
            logging.exception(
                "Error occurred during adversarial loss calculation:")
            # Return zero loss if adversarial generation fails
            return torch.tensor(0.0, device=data.device)

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

                    # Generate adversarial examples for current batch
                    orig, adv_data, _ = self.attack.attack(data, target)

                    # Move results to CPU and convert to numpy to save memory
                    all_originals.append(orig.cpu())
                    all_adversarials.append(adv_data.cpu())
                    all_labels.append(target.cpu())

                    total_processed += len(data)

                    # Clear GPU memory
                    del data, target, orig, adv_data
                    torch.cuda.empty_cache()

                    if batch_idx % 10 == 0:
                        logging.info(f"Processed {total_processed} samples...")

                    # Save intermediate results if memory usage is high
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
        try:
            # Determine folder structure similar to Trainer.save_model
            task = getattr(self.config, 'task_name', 'default_task')

            # Handle dataset name properly - convert list to string if needed
            if hasattr(self.config, 'data_key'):
                dataset = self.config.data_key
            elif hasattr(self.config, 'data'):
                if isinstance(self.config.data, list):
                    # Take first dataset if it's a list
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

            # Handle attack name - convert list to string if needed
            if isinstance(self.config.attack_name, list):
                # Join attack names with +
                attack = "+".join(self.config.attack_name)
            else:
                attack = self.config.attack_name

            folder = os.path.join("out", task, dataset,
                                  model_name, "attack", attack)
            os.makedirs(folder, exist_ok=True)

            # The rest of the method remains the same
            num_samples = min(5, adv_data.size(0))
            for i in range(num_samples):
                orig_filename = os.path.join(folder, f"sample_{i}_orig.png")
                adv_filename = os.path.join(folder, f"sample_{i}_adv.png")
                pert_filename = os.path.join(
                    folder, f"sample_{i}_perturbation.png")
                save_image(orig[i], orig_filename)
                save_image(adv_data[i], adv_filename)
                perturbation = adv_data[i] - orig[i]
                save_image(perturbation, pert_filename)

            perturbation_tensor = adv_data[:num_samples] - orig[:num_samples]
            perturbations = perturbation_tensor.view(num_samples, -1)
            avg_norm = torch.norm(perturbations, p=2, dim=1).mean().item()
            summary_path = os.path.join(folder, "summary.txt")
            # Specify UTF-8 encoding to handle Unicode characters correctly.
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("Attack Summary\n")
                f.write("======================\n")
                f.write(f"Attack Type: {attack}\n")
                f.write(f"Epsilon: {self.config.epsilon}\n")
                if hasattr(self.config, 'attack_alpha'):
                    f.write(f"Attack Alpha: {self.config.attack_alpha}\n")
                if hasattr(self.config, 'attack_steps'):
                    f.write(f"Attack Steps: {self.config.attack_steps}\n")
                f.write(f"Number of samples saved: {num_samples}\n")
                f.write(f"Average Perturbation ℓ₂ Norm: {avg_norm:.4f}\n")
            logging.info(
                f"Saved {num_samples} adversarial samples and summary to {folder}")
        except Exception as e:
            logging.exception("Exception in save_attack_samples:")

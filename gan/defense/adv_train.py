# adv_train.py

import torch
from gan.attack.attack_loader import AttackLoader
from gan.attack.attack_data_loader import AttackDataLoader
import logging
import os
from torchvision.utils import save_image
import gc


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

        # Initialize the attack using AttackLoader
        self.attack_loader = AttackLoader(model, config)
        self.attack = self.attack_loader.get_attack(config.attack_name)
        
        # Log the attack type(s) being used
        if isinstance(config.attack_name, list):
            logging.info(f"Setting up adversarial training with attacks: {', '.join(config.attack_name)}")
        else:
            logging.info(f"Setting up adversarial training with attack: {config.attack_name}")

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

    def adversarial_loss(self, data, target, batch_indices=None):
        try:
            if self.use_pregenerated and batch_indices is not None:
                # Ensure batch_indices is a tensor
                if not isinstance(batch_indices, torch.Tensor):
                    batch_indices = torch.tensor(batch_indices)

                # Modulo operation to wrap indices within available attacks
                if self.attack_data.data.get('train') is not None:
                    max_idx = self.attack_data.data['train']['original'].size(
                        0)
                    batch_indices = batch_indices % max_idx

                # Try to load pre-generated attacks
                orig, adv_data, _ = self.attack_data.get_attack_batch(
                    batch_indices)

                if orig is not None and adv_data is not None:
                    orig = orig.to(data.device)
                    adv_data = adv_data.to(data.device)
                    logging.debug(
                        f"Using pre-generated attacks (indices {batch_indices[0]}-{batch_indices[-1]})")
                else:
                    # Fall back to generating attacks if loading fails
                    orig, adv_data, _ = self.attack.attack(data, target)
            else:
                # Generate attacks on-the-fly if no pre-generated attacks available
                orig, adv_data, _ = self.attack.attack(data, target)

            # Save samples only once
            if not self._attack_samples_saved:
                self.save_attack_samples(orig, adv_data)
                self._attack_samples_saved = True
            with torch.cuda.amp.autocast():
                adv_output = self.model(adv_data)
                adv_loss = self.criterion(adv_output, target)
            return adv_loss
        except Exception as e:
            logging.exception(
                "Error occurred during adversarial loss calculation:")
            # Return zero loss if adversarial generation fails
            return torch.tensor(0.0).to(data.device)

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
                    dataset = self.config.data[0]  # Take first dataset if it's a list
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
                attack = "+".join(self.config.attack_name)  # Join attack names with +
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

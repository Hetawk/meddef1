import torch
import os
import logging
from pathlib import Path
from tqdm import tqdm
import gc
from PIL import Image
from torchvision import transforms


class AttackDataLoader:
    def __init__(self, dataset_name, model_name, attack_type, device=None, max_batch_size=32):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.attack_type = attack_type
        self.attacks_path = Path("out/attacks")
        self.data = {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_batch_size = max_batch_size
        self.current_batch = None
        self.transform = transforms.ToTensor()  # Add ToTensor transform
        logging.info(f"AttackDataLoader initialized using {self.device}")
        logging.info(
            f"AttackDataLoader initialized with max batch size: {max_batch_size}")

    def validate_attacks_exist(self, split="train"):
        """Check if pre-generated attacks exist and log percentage of dataset covered"""
        attack_dir = self.attacks_path / self.dataset_name / \
            self.model_name / self.attack_type / split
        adv_dir = attack_dir / "adversarial"

        if not adv_dir.exists():
            return False

        try:
            # Load metadata to check coverage
            metadata = torch.load(str(attack_dir / "metadata.pt"))
            total_samples = metadata.get('total_samples', 0)
            attacked_samples = metadata.get('attacked_samples', 0)
            coverage = (attacked_samples / total_samples *
                        100) if total_samples > 0 else 0
            logging.info(
                f"Found pre-generated attacks covering {coverage:.2f}% of {split} dataset")
            return True
        except Exception as e:
            logging.warning(f"Could not verify attack coverage: {str(e)}")
            return False

    def load_attacks(self, split="train"):
        """Load pre-generated attacks from image directories"""
        if split in self.data:
            return True

        attack_dir = self.attacks_path / self.dataset_name / \
            self.model_name / self.attack_type / split
        adv_dir = attack_dir / "adversarial"

        if not adv_dir.exists():
            logging.warning(
                f"No pre-generated attacks found at {attack_dir}")
            return False

        try:
            logging.info(f"Loading {split} attacks from {attack_dir}")

            # Load metadata
            metadata = torch.load(str(attack_dir / "metadata.pt"))
            total_samples = metadata.get('total_samples', 0)
            attacked_samples = metadata.get('attacked_samples', 0)

            # List all image files
            adv_files = sorted(os.listdir(adv_dir))

            # Ensure the number of files matches the metadata
            if len(adv_files) != attacked_samples:
                logging.error(
                    f"Mismatch in number of attack files and metadata: {len(adv_files)} vs {attacked_samples}")
                return False

            # Store file paths and metadata
            self.data[split] = {
                'adv_dir': adv_dir,
                'adv_files': adv_files,
                'total_samples': total_samples,
                'attacked_samples': attacked_samples
            }
            logging.info(
                f"Loaded {attacked_samples} attacks from {attack_dir}")
            return True

        except Exception as e:
            logging.error(f"Error loading attacks: {str(e)}")
            return False

    def get_attack_batch(self, indices, split="train"):
        """Get a batch of attacks by indices from image files"""
        try:
            if split not in self.data and not self.load_attacks(split):
                return None, None, None

            data = self.data[split]
            adv_dir = data['adv_dir']
            adv_files = data['adv_files']

            # Load images and convert to tensors
            adversarial_images = []

            for i in indices:
                adv_path = os.path.join(adv_dir, adv_files[i])

                try:
                    adv_image = Image.open(adv_path).convert('RGB')
                    adv_tensor = self.transform(adv_image).to(self.device)
                    adversarial_images.append(adv_tensor)

                    adv_image.close()

                except Exception as e:
                    logging.error(f"Error loading image: {str(e)}")
                    return None, None, None

            # Stack images
            adversarial_images = torch.stack(adversarial_images)

            # Dummy original images and labels
            original_images = torch.zeros_like(adversarial_images)
            labels = torch.zeros(
                len(indices), dtype=torch.long).to(self.device)

            return original_images, adversarial_images, labels

        except Exception as e:
            logging.error(f"Error retrieving attack batch: {str(e)}")
            logging.error(
                f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            return None, None, None

    def get_coverage_stats(self, split="train"):
        """Get statistics about attack coverage"""
        try:
            attack_dir = self.attacks_path / self.dataset_name / \
                self.model_name / self.attack_type / split
            metadata_path = attack_dir / "metadata.pt"
            if metadata_path.exists():
                metadata = torch.load(str(metadata_path))
                return {
                    'total_samples': metadata.get('total_samples', 0),
                    'covered_samples': metadata.get('attacked_samples', 0),
                    'coverage_percent': (metadata.get('attacked_samples', 0) / metadata.get('total_samples', 0)) * 100,
                    'batch_size': metadata.get('batch_size', 0)
                }
        except Exception as e:
            logging.error(f"Error loading coverage stats: {str(e)}")
        return None

    def cleanup(self):
        """Clean up memory"""
        self.data.clear()
        if self.current_batch is not None:
            del self.current_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

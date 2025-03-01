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
        
        # Handle attack_type whether it's a string or a list
        if isinstance(attack_type, list):
            # Join multiple attack names with '+' for directory naming
            self.attack_type = "+".join(attack_type)
            logging.info(f"Using composite attack type: {self.attack_type} for data loading")
        else:
            self.attack_type = attack_type
            
        self.data = {}
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"AttackDataLoader initialized using {self.device}")
        
        self.max_batch_size = 32
        logging.info(f"AttackDataLoader initialized with max batch size: {self.max_batch_size}")

    def get_attack_path(self, split='train'):
        """Get path to stored adversarial examples for a specific split"""
        base_path = Path('out') / 'attacks' / self.dataset_name / self.model_name / self.attack_type / split
        return base_path

    def validate_attacks_exist(self, split='train'):
        """Check if pre-generated attacks exist"""
        try:
            path = self.get_attack_path(split)
            if not path.exists():
                return False
            
            attack_file = path / 'attacks.pt'
            if not attack_file.exists():
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error checking attack existence: {e}")
            return False

    def load_attacks(self, split='train'):
        """Load pre-generated attacks"""
        try:
            path = self.get_attack_path(split)
            attack_file = path / 'attacks.pt'
            
            if not attack_file.exists():
                logging.warning(f"No pre-generated attacks found at {attack_file}")
                return False
                
            logging.info(f"Loading pre-generated attacks from {attack_file}")
            self.data[split] = torch.load(attack_file, map_location=self.device)
            
            # Validate data shape
            for key in ['original', 'adversarial', 'labels']:
                if key not in self.data[split]:
                    logging.error(f"Missing key {key} in loaded attack data")
                    return False
                
            logging.info(f"Loaded {self.data[split]['original'].size(0)} {split} attack samples")
            return True
        except Exception as e:
            logging.error(f"Error loading attacks: {e}")
            return False

    def get_attack_batch(self, indices, split='train'):
        """Get specific batch of attack data"""
        try:
            # Ensure data is loaded
            if split not in self.data:
                loaded = self.load_attacks(split)
                if not loaded:
                    return None, None, None
            
            # Get data for indices
            orig = self.data[split]['original'][indices]
            adv = self.data[split]['adversarial'][indices]
            labels = self.data[split]['labels'][indices]
            
            return orig, adv, labels
        except Exception as e:
            logging.error(f"Error retrieving attack batch: {e}")
            return None, None, None

    def save_attacks(self, original, adversarial, labels, split='train'):
        """Save generated attacks"""
        try:
            path = self.get_attack_path(split)
            os.makedirs(path, exist_ok=True)
            
            attack_file = path / 'attacks.pt'
            
            torch.save({
                'original': original,
                'adversarial': adversarial,
                'labels': labels
            }, attack_file)
            
            logging.info(f"Saved {len(original)} attack samples to {attack_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving attacks: {e}")
            return False

    def get_coverage_stats(self, split="train"):
        """Get statistics about attack coverage"""
        try:
            attack_dir = self.get_attack_path(split)
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

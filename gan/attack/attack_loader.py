import logging
import torch
from tqdm import tqdm

# Import all attack implementations
from gan.attack.fgsm import FGSMAttack
from gan.attack.pgd import PGDAttack
from gan.attack.jsma import JSMAAttack
from gan.attack.bim import BIMAttack
from gan.attack.cw import CWAttack
from gan.attack.zoo import ZooAttack
from gan.attack.boundary import BoundaryAttack
from gan.attack.elasticnet import ElasticNetAttack
from gan.attack.onepixel import OnePixelAttack


class AttackLoader:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Ensure attack_name is set from config (similar to AdversarialTraining)
        if not hasattr(config, 'attack_name'):
            self.config.attack_name = getattr(config, 'attack_type', 'fgsm')

        # Supported attacks with their own implementations
        self.supported_attacks = {
            'fgsm': FGSMAttack,
            'pgd': PGDAttack,
            'bim': BIMAttack,
            'jsma': JSMAAttack,
            'cw': CWAttack,
            'zoo': ZooAttack,
            'boundary': BoundaryAttack,
            'elasticnet': ElasticNetAttack,
            'onepixel': OnePixelAttack
        }

    def get_attack(self, attack_name):
        try:
            key = attack_name.lower()
            if key not in self.supported_attacks:
                logging.error(f"Attack {attack_name} not supported")
                return None

            # Get epsilon from config, with proper fallback
            epsilon = getattr(self.config, 'epsilon', None)
            if epsilon is None:
                epsilon = getattr(self.config, 'attack_eps', 0.3)

            # Handle different types of attacks with appropriate parameters
            if key in ['pgd', 'bim']:
                alpha = getattr(self.config, 'attack_alpha', 0.01)
                steps = getattr(self.config, 'attack_steps', 40)
                return self.supported_attacks[key](self.model, epsilon, alpha, steps)
            
            elif key == 'cw':
                c = getattr(self.config, 'attack_c', 1.0)
                iterations = getattr(self.config, 'attack_iterations', 100)
                lr = getattr(self.config, 'attack_lr', 0.01)
                binary_search_steps = getattr(self.config, 'attack_binary_steps', 9)
                confidence = getattr(self.config, 'attack_confidence', 0)
                return self.supported_attacks[key](self.model, epsilon, c, iterations, lr, binary_search_steps, confidence)
            
            elif key == 'zoo':
                iterations = getattr(self.config, 'attack_iterations', 100)
                h = getattr(self.config, 'attack_h', 0.001)
                binary_search_steps = getattr(self.config, 'attack_binary_steps', 5)
                return self.supported_attacks[key](self.model, epsilon, iterations, h, binary_search_steps)
                
            elif key == 'boundary':
                steps = getattr(self.config, 'attack_steps', 50)
                spherical_step = getattr(self.config, 'attack_spherical_step', 0.01)
                source_step = getattr(self.config, 'attack_source_step', 0.01)
                step_adaptation = getattr(self.config, 'attack_step_adaptation', 1.5)
                max_directions = getattr(self.config, 'attack_max_directions', 25)
                return self.supported_attacks[key](self.model, epsilon, steps, spherical_step, source_step, step_adaptation, max_directions)
                
            elif key == 'elasticnet':
                alpha = getattr(self.config, 'attack_alpha', 0.01)
                iterations = getattr(self.config, 'attack_iterations', 40)
                beta = getattr(self.config, 'attack_beta', 1.0)
                return self.supported_attacks[key](self.model, epsilon, alpha, iterations, beta)
                
            elif key == 'onepixel':
                pixel_count = getattr(self.config, 'attack_pixel_count', 1)
                max_iter = getattr(self.config, 'attack_max_iter', 100)
                popsize = getattr(self.config, 'attack_popsize', 10)
                return self.supported_attacks[key](self.model, epsilon, pixel_count, max_iter, popsize)
                
            else:
                # For FGSM and JSMA
                return self.supported_attacks[key](self.model, epsilon)

        except Exception as e:
            logging.exception(f"Error initializing attack {attack_name}:")
            return None


class AttackHandler:
    def __init__(self, model, attack_name, args):
        self.device = next(model.parameters()).device
        self.attack_loader = AttackLoader(model, args)
        self.attack = self.attack_loader.get_attack(attack_name)

    def generate_adversarial_samples_batch(self, images, labels):
        """Generate adversarial examples for a single batch"""
        images = images.to(self.device)
        labels = labels.to(self.device)

        orig, adv, labs = self.attack.attack(images, labels)

        return {
            'original': orig.cpu(),
            'adversarial': adv.cpu(),
            'labels': labs.cpu()
        }

    def generate_adversarial_samples(self, data_loader):
        """Full dataset attack generation with progress bar"""
        results = {'original': [], 'adversarial': [], 'labels': []}
        pbar = tqdm(data_loader,
                    desc=f"Generating {self.attack_loader.config.attack_name} attacks (untargeted)",
                    unit="batch")

        for images, labels in pbar:
            batch_results = self.generate_adversarial_samples_batch(
                images, labels)
            results['original'].append(batch_results['original'])
            results['adversarial'].append(batch_results['adversarial'])
            results['labels'].append(batch_results['labels'])
            pbar.set_postfix({'batch_size': images.size(0)})

        pbar.close()
        return results

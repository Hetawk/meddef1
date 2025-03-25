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


class CompositeAttack:
    """A wrapper for multiple attack methods that can be used interchangeably"""

    def __init__(self, attacks):
        self.attacks = attacks
        self.current_attack_idx = 0

    def attack(self, images, labels):
        """Rotates through available attacks for each call"""
        attack = self.attacks[self.current_attack_idx]
        self.current_attack_idx = (
            self.current_attack_idx + 1) % len(self.attacks)
        return attack.attack(images, labels)

    def generate(self, images, labels, epsilon=None):
        """If attacks support generate method, use it"""
        attack = self.attacks[self.current_attack_idx]
        self.current_attack_idx = (
            self.current_attack_idx + 1) % len(self.attacks)
        if hasattr(attack, 'generate'):
            return attack.generate(images, labels, epsilon)
        else:
            # Fallback to attack method
            _, adv_images, _ = attack.attack(images, labels)
            return adv_images


class AttackLoader:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Ensure attack_name is set from config, handling both string and list
        if not hasattr(config, 'attack_name'):
            attack_type = getattr(config, 'attack_type', ['fgsm'])
            # Handle both string and list formats
            if isinstance(attack_type, str):
                self.config.attack_name = [attack_type]
            else:
                self.config.attack_name = attack_type

        # Supported attacks with their own implementations
        self.supported_attacks = {
            'fgsm': FGSMAttack,
            'pgd': self._get_pgd_attack,  # Special handler for PGD to use enhanced version
            'bim': BIMAttack,
            'jsma': JSMAAttack,
            'cw': CWAttack,
            'zoo': ZooAttack,
            'boundary': BoundaryAttack,
            'elasticnet': ElasticNetAttack,
            'onepixel': OnePixelAttack
        }

    def _get_pgd_attack(self, *args, **kwargs):
        """Special handler for PGD to use the enhanced implementation"""
        try:
            # Use PGDAttack consistently
            return PGDAttack(self.model, self.config)
        except Exception as e:
            logging.warning(
                f"Error initializing PGDAttack: {e}. Falling back to original implementation.")
            # Fall back to original implementation if there's an issue
            return PGDAttack(*args, **kwargs)

    def get_attack(self, attack_names):
        """
        Initialize one or more attack methods

        Args:
            attack_names: str or list of str - name(s) of attacks to initialize

        Returns:
            Single attack object or CompositeAttack for multiple attacks
        """
        # Convert to list if it's a string
        if isinstance(attack_names, str):
            attack_names = [attack_names]

        attacks = []

        for attack_name in attack_names:
            try:
                key = attack_name.lower()
                if key not in self.supported_attacks:
                    logging.error(f"Attack {attack_name} not supported")
                    continue

                # Get epsilon from config, with proper fallback
                epsilon = getattr(self.config, 'epsilon', None)
                if epsilon is None:
                    epsilon = getattr(self.config, 'attack_eps', 0.3)

                # Handle different types of attacks with appropriate parameters
                if key == 'pgd':
                    # Special handling for PGD - use the function that decides which implementation to use
                    alpha = getattr(self.config, 'attack_alpha', 0.01)
                    steps = getattr(self.config, 'attack_steps', 40)
                    attack_handler = self.supported_attacks[key]
                    attack = attack_handler(self.model, epsilon, alpha, steps)
                    attacks.append(attack)
                elif key == 'bim':
                    alpha = getattr(self.config, 'attack_alpha', 0.01)
                    steps = getattr(self.config, 'attack_steps', 40)
                    attacks.append(self.supported_attacks[key](
                        self.model, epsilon, alpha, steps))

                elif key == 'cw':
                    c = getattr(self.config, 'attack_c', 1.0)
                    # Use the provided iterations value
                    iterations = getattr(self.config, 'iterations', 40)
                    lr = getattr(self.config, 'attack_lr', 0.01)
                    binary_search_steps = getattr(
                        self.config, 'attack_binary_steps', 9)
                    confidence = getattr(self.config, 'attack_confidence', 0)
                    attacks.append(self.supported_attacks[key](
                        self.model, epsilon, c, iterations, lr, binary_search_steps, confidence))

                elif key == 'zoo':
                    # Use the provided iterations value
                    iterations = getattr(self.config, 'iterations', 40)
                    h = getattr(self.config, 'attack_h', 0.001)
                    binary_search_steps = getattr(
                        self.config, 'attack_binary_steps', 5)
                    attacks.append(self.supported_attacks[key](
                        self.model, epsilon, iterations, h, binary_search_steps))

                elif key == 'boundary':
                    # Use 40 as default as specified
                    steps = getattr(self.config, 'attack_steps', 40)
                    spherical_step = getattr(
                        self.config, 'attack_spherical_step', 0.01)
                    source_step = getattr(
                        self.config, 'attack_source_step', 0.01)
                    step_adaptation = getattr(
                        self.config, 'attack_step_adaptation', 1.5)
                    max_directions = getattr(
                        self.config, 'attack_max_directions', 25)
                    attacks.append(self.supported_attacks[key](
                        self.model, epsilon, steps, spherical_step, source_step, step_adaptation, max_directions))

                elif key == 'elasticnet':
                    alpha = getattr(self.config, 'attack_alpha', 0.01)
                    # Use the provided iterations value
                    iterations = getattr(self.config, 'iterations', 40)
                    beta = getattr(self.config, 'attack_beta', 1.0)
                    attacks.append(self.supported_attacks[key](
                        self.model, epsilon, alpha, iterations, beta))

                elif key == 'onepixel':
                    pixel_count = getattr(self.config, 'attack_pixel_count', 1)
                    # Use the provided iterations value
                    max_iter = getattr(self.config, 'iterations', 40)
                    popsize = getattr(self.config, 'attack_popsize', 10)
                    attacks.append(self.supported_attacks[key](
                        self.model, epsilon, pixel_count, max_iter, popsize))

                else:
                    # For FGSM and JSMA
                    attacks.append(
                        self.supported_attacks[key](self.model, epsilon))

            except Exception as e:
                logging.exception(f"Error initializing attack {attack_name}:")

        if not attacks:
            logging.error(
                f"Could not initialize any of the requested attacks: {attack_names}")
            return None

        if len(attacks) == 1:
            logging.info(f"Initialized {attack_names[0]} attack")
            return attacks[0]
        else:
            logging.info(
                f"Initialized composite attack with: {', '.join(attack_names)}")
            return CompositeAttack(attacks)


class AttackHandler:
    def __init__(self, model, attack_name, args):
        self.device = next(model.parameters()).device
        self.attack_loader = AttackLoader(model, args)
        self.attack = self.attack_loader.get_attack(attack_name)

        # Store attack name for logging
        self.attack_name = attack_name
        if isinstance(attack_name, list):
            self.attack_name_str = '+'.join(attack_name)
        else:
            self.attack_name_str = attack_name

    def generate_adversarial_samples_batch(self, images, labels):
        """Generate adversarial examples for a single batch"""
        images = images.to(self.device)
        labels = labels.to(self.device)

        orig, adv, labs = self.attack.attack(images, labels)

        return {
            'original': orig.cpu(),
            'adversarial': adv.cpu(),
            'labels': labs.cpu() if labs is not None else labels.cpu()
        }

    def generate_adversarial_samples(self, data_loader):
        """Full dataset attack generation with progress bar"""
        results = {'original': [], 'adversarial': [], 'labels': []}
        pbar = tqdm(data_loader,
                    desc=f"Generating {self.attack_name_str} attacks (untargeted)",
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

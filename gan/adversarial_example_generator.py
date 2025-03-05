import os
from gan.gan import GAN
from gan.attack.attack_loader import AttackLoader
from utils.visual.visualization import Visualization
import torch
import logging


class AdversarialExampleGenerator:
    def __init__(self, noise_dim, data_dim, device, model, args):
        self.gan = GAN(noise_dim=noise_dim, data_dim=data_dim, device=device)
        self.device = device
        self.visualization = Visualization()
        self.attack_loader = AttackLoader(model, args)

    def generate_adversarial_examples(self, batch_size):
        noise = torch.randn(
            batch_size, self.gan.generator.model[0].in_features).to(self.device)
        adversarial_examples = self.gan.generate(noise)
        logging.info(f"Generated {batch_size} adversarial examples.")

        # Check if the output is a flattened tensor that needs reshaping
        if adversarial_examples.dim() == 2 and adversarial_examples.size(1) == 3 * 224 * 224:
            adversarial_examples = adversarial_examples.view(-1, 3, 224, 224)
        # Accept if already in image shape
        elif adversarial_examples.dim() == 4 and adversarial_examples.size(1) == 3:
            pass
        else:
            logging.error(
                f"Unexpected shape for adversarial examples: {adversarial_examples.shape}")
        return adversarial_examples

    def generate_adversarial_examples_with_attack(self, data_loader, attack_name):
        attack = self.attack_loader.get_attack(attack_name)
        results = {'original': [], 'adversarial': [], 'labels': []}

        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            orig_images, adv_images, labels = attack.attack(images, labels)
            results['original'].append(orig_images.cpu())
            results['adversarial'].append(adv_images.cpu())
            results['labels'].append(labels.cpu())

        logging.info(
            f"Generated adversarial examples using {attack_name} attack.")
        return results

    def visualize_adversarial_examples(self, original_data, adversarial_examples, model_name, task_name, dataset_name,
                                       attack_name):
        self.visualization.visualize_attack(original_data, adversarial_examples, None, model_name, task_name,
                                            dataset_name, attack_name)
        logging.info(
            f"Adversarial examples visualized and saved for model {model_name}.")

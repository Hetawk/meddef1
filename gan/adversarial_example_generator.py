# adversarial_example_generator.py
import os
import torch
import logging
from gan.gan import GAN
from utils.visual.visualization import Visualization

class AdversarialExampleGenerator:
    def __init__(self, noise_dim, data_dim, device):
        self.gan = GAN(noise_dim=noise_dim, data_dim=data_dim, device=device)
        self.device = device
        self.visualization = Visualization()

    def generate_adversarial_examples(self, batch_size):
        noise = torch.randn(batch_size, self.gan.generator.model[0].in_features).to(self.device)
        adversarial_examples = self.gan.generate(noise)
        logging.info(f"Generated {batch_size} adversarial examples.")
        return adversarial_examples

    def visualize_adversarial_examples(self, original_data, adversarial_examples, model_name, task_name, dataset_name, attack_name):
        adv_examples = (original_data, adversarial_examples)
        model_names = [model_name]
        self.visualization.visualize_attack(original_data, adversarial_examples, None, model_name, task_name, dataset_name, attack_name)
        logging.info(f"Adversarial examples visualized and saved for model {model_name}.")
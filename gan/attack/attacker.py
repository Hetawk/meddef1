# attacker.py
import logging
import torch

from gan.attack.attack_loader import AttackLoader

class AttackHandler:
    def __init__(self, model, attack_name, **attack_params):
        """
        Initialize the AttackHandler with a model, attack name, and parameters.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            attack_name (str): Name of the attack to use (e.g., 'fgsm', 'pgd').
            attack_params (dict): Parameters for the attack (e.g., epsilon for FGSM).
        """
        self.device = next(model.parameters()).device
        self.attack_loader = AttackLoader(model)
        self.attack = self.attack_loader.get_attack(attack_name, **attack_params)

    def generate_adversarial_samples(self, data_loader):
        """
        Generate adversarial samples for each batch in data_loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to attack.

        Returns:
            dict: A dictionary with keys 'original', 'adversarial', and 'labels' containing lists of samples.
        """
        results = {'original': [], 'adversarial': [], 'labels': []}

        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            orig_images, adv_images, labels = self.attack.attack(images, labels)
            results['original'].append(orig_images.cpu())
            results['adversarial'].append(adv_images.cpu())
            results['labels'].append(labels.cpu())

        logging.info("Generated adversarial samples for all batches.")
        return results

    def generate_adv_examples(self, test_loader):
        """
        Generate adversarial examples for each attack in the attack loader.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: A dictionary with attack names as keys and tuples of adversarial examples and labels as values.
        """
        adv_examples_dict = {}
        for attack_name in self.attack_loader.attacks_dict.keys():
            logging.info(f"Running attack: {attack_name}")
            attack = self.attack_loader.get_attack(attack_name)

            adv_examples_list = []
            adv_labels_list = []
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                orig_images, adv_images, labels = attack.attack(data, labels)

                if adv_images is not None:
                    adv_examples_list.append(adv_images.cpu())
                    adv_labels_list.append(labels.cpu())

            adv_examples_dict[attack_name] = (adv_examples_list, adv_labels_list)
        return adv_examples_dict

    def compute_perturbation_data(self, adv_examples_dict):
        """
        Compute the perturbation data for each adversarial example.

        Args:
            adv_examples_dict (dict): Dictionary with attack names as keys and tuples of adversarial examples and labels as values.

        Returns:
            dict: A dictionary with attack names as keys and lists of perturbation values as values.
        """
        perturbations = {}
        for attack_name, (adv_examples, _) in adv_examples_dict.items():
            perturbation_list = []
            for example in adv_examples:
                perturbation = example - example.detach()
                perturbation_list.append(perturbation.abs().mean().item())
            perturbations[attack_name] = perturbation_list
        return perturbations

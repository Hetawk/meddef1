# feat_denoising.py
import torch
import torch.nn as nn


class FeatDenoising:
    def __init__(self, model, noise_std=0.1):
        self.model = model
        self.noise_std = noise_std
        self.criterion = nn.CrossEntropyLoss()

    def apply_noise(self, features):
        """
        Add Gaussian noise to the features.
        """
        noise = torch.normal(mean=0, std=self.noise_std, size=features.size()).to(features.device)
        return features + noise

    def extract_features(self, x):
        """
        Extract intermediate features from the model.
        """
        # Assuming the model has a feature extraction method. You may need to adjust this based on your model architecture.
        self.model.eval()
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features

    def defend(self, adv_examples, adv_labels):
        """
        Apply feature denoising to adversarial examples and evaluate the model's performance.
        """
        features = self.extract_features(adv_examples)
        noisy_features = self.apply_noise(features)

        # Reconstruct the input from noisy features and run the model
        # Assuming a function to reconstruct input from features is available
        # If not, you may need to modify your model to handle this
        noisy_examples = self.model.reconstruct_from_features(noisy_features)

        with torch.no_grad():
            outputs = self.model(noisy_examples)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == adv_labels).sum().item()

        return self.model, correct, adv_examples.size(0)

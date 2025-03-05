# feat_squeeze.py
import torch
import torch.nn as nn


class FeatSqueeze:
    def __init__(self, model, squeeze_factor=0.1):
        self.model = model
        self.squeeze_factor = squeeze_factor
        self.criterion = nn.CrossEntropyLoss()

    def squeeze_features(self, features):
        """
        Apply feature squeezing by reducing the precision of the features.
        """
        # Example: Apply quantization by reducing the precision of the features
        squeezed_features = (features / self.squeeze_factor).round() * self.squeeze_factor
        return squeezed_features

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
        Apply feature squeezing to adversarial examples and evaluate the model's performance.
        """
        features = self.extract_features(adv_examples)
        squeezed_features = self.squeeze_features(features)

        # Reconstruct the input from squeezed features and run the model
        # Assuming a function to reconstruct input from features is available
        # If not, you may need to modify your model to handle this
        squeezed_examples = self.model.reconstruct_from_features(squeezed_features)

        with torch.no_grad():
            outputs = self.model(squeezed_examples)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == adv_labels).sum().item()

        return self.model, correct, adv_examples.size(0)

# adv_train.py

import torch

class AdversarialTraining:
    def __init__(self, model, criterion, epsilon=0.3, alpha=0.01):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Step size for gradient ascent

    def generate_adversarial_example(self, data, target):
        data.requires_grad = True
        output = self.model(data)
        loss = self.criterion(output, target)
        self.model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = data + self.alpha * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data

    def adversarial_loss(self, data, target):
        adv_data = self.generate_adversarial_example(data, target)
        adv_output = self.model(adv_data)
        adv_loss = self.criterion(adv_output, target)
        return adv_loss

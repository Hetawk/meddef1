# adv_train.py

import torch
from gan.attack.attack_loader import AttackLoader


class AdversarialTraining:
    def __init__(self, model, criterion, args):
        self.model = model
        self.criterion = criterion
        self.args = args
        # Instantiate AttackLoader and get the desired attack
        self.attack_loader = AttackLoader(model, args)
        self.attack = self.attack_loader.get_attack(args.attack_name)

    def adversarial_loss(self, data, target):
        # Use the attack to generate adversarial examples
        adv_data = self.attack.generate(data, target)
        adv_output = self.model(adv_data)
        adv_loss = self.criterion(adv_output, target)
        return adv_loss

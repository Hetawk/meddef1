import logging
from gan.attack.fgsm import FGSMAttack
from gan.attack.pgd import PGDAttack
from gan.attack.boundary import BoundaryAttack
from gan.attack.bim import BIMAttack
from gan.attack.cw import CWAttack
from gan.attack.elasticnet import ElasticNetAttack
from gan.attack.jsma import JSMAAttack
from gan.attack.onepixel import OnePixelAttack
from gan.attack.zoo import ZooAttack

class AttackLoader:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.attacks_dict = {
            'fgsm': FGSMAttack(self.model, epsilon=args.epsilon),
            'pgd': PGDAttack(self.model, epsilon=args.epsilon, alpha=args.alpha, iterations=args.iterations),
            'bim': BIMAttack(self.model, epsilon=args.epsilon, alpha=args.alpha, iterations=args.iterations),
            # 'cw': CWAttack(self.model),
            # 'elasticnet': ElasticNetAttack(self.model),
            # 'boundary': BoundaryAttack(self.model),
            # 'jsma': JSMAAttack(self.model),
            # 'onepixel': OnePixelAttack(self.model),
            # 'zoo': ZooAttack(self.model)
        }
        logging.info("AttackLoader initialized with attacks: " + ", ".join(self.attacks_dict.keys()))

    def get_attack(self, attack_name):
        logging.info(f"Getting attack {attack_name}.")
        if attack_name in self.attacks_dict:
            return self.attacks_dict[attack_name]
        else:
            raise ValueError(f"Attack {attack_name} not recognized.")
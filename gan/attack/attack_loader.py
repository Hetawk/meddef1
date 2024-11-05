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
    def __init__(self, model):
        self.model = model
        self.attacks_dict = {
            'fgsm': FGSMAttack,
            'pgd': PGDAttack,
            'boundary': BoundaryAttack,
            'bim': BIMAttack,
            'cw': CWAttack,
            'elasticnet': ElasticNetAttack,
            'jsma': JSMAAttack,
            'onepixel': OnePixelAttack,
            'zoo': ZooAttack
        }
        logging.info("AttackLoader initialized with attacks: " + ", ".join(self.attacks_dict.keys()))

    def get_attack(self, attack_name, **attack_params):
        logging.info(f"Getting attack {attack_name}.")
        if attack_name in self.attacks_dict:
            return self.attacks_dict[attack_name](self.model, **attack_params)
        else:
            raise ValueError(f"Attack {attack_name} not recognized.")
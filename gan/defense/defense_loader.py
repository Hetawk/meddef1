# # defense_loader.py
# from .adv_train import AdversarialTraining
# from .cert_defense import CertDefense
# from .def_distill import DefDistill
# from .ensemble_adv_train import EnsembleAdvTrain
# from .feat_denoising import FeatDenoising
# from .feat_squeeze import FeatSqueeze
# from .grad_mask import GradMask
# from .randomization import Randomization
# import logging

# class DefenseLoader:
#     def __init__(self, model):
#         self.model = model
#         self.defenses_dict = {
#             'adv_train': AdversarialTraining,
#             'cert_defense': CertDefense,
#             'def_distill': DefDistill,
#             'ensemble_adv_train': EnsembleAdvTrain,
#             'feat_denoising': FeatDenoising,
#             'feat_squeeze': FeatSqueeze,
#             'grad_mask': GradMask,
#             'randomization': Randomization,
#             # Add more defenses here as needed
#         }
#         logging.info("DefenseLoader initialized with defenses: " + ", ".join(self.defenses_dict.keys()))

#     def get_defense(self, defense_name, **kwargs):
#         logging.info(f"Getting defense {defense_name}.")
#         if defense_name in self.defenses_dict:
#             defense_class = self.defenses_dict[defense_name]
#             defense = defense_class(self.model, **kwargs)
#             return defense
#         else:
#             raise ValueError(f"Defense {defense_name} not recognized.")

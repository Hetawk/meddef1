import torch
import torch.nn.utils.prune as prune
import logging

class Pruner:
    def __init__(self, model, prune_rate):
        self.model = model
        self.prune_rate = prune_rate

    def unstructured_prune(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.prune_rate)
                logging.info(f"Pruned {name} with rate {self.prune_rate}")
        return self.model

    def save_checkpoint(self, checkpoint, filename='checkpoint.pth.tar'):
        torch.save(checkpoint, filename)
        logging.info(f'Checkpoint saved to {filename}')
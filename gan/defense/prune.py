# pruner.py is a module for pruning neural networks using L1 unstructured pruning. The Pruner class in this module takes a model and a pruning rate as input and applies L1 unstructured pruning to the weights of Conv2d and Linear layers in the model. The unstructured_prune method iterates over all named modules in the model and prunes the weight parameter of Conv2d and Linear layers using the prune.l1_unstructured function. The save_checkpoint method saves the pruned model checkpoint to a file. The logging module is used to log messages to the console.

import torch
import torch.nn.utils.prune as prune
import logging

class Pruner:
    def __init__(self, model, prune_rate):
        self.model = model
        self.prune_rate = prune_rate

    def unstructured_prune(self):
        # Iterate over all named modules in the model
        for name, module in self.model.named_modules():
            # Check if the module is a Conv2d or Linear layer
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # Apply L1 unstructured pruning to the weight parameter of the module
                # L1 unstructured pruning is a type of magnitude-based pruning where
                # weights with the smallest absolute values (L1 norm) are pruned
                prune.l1_unstructured(module, name='weight', amount=self.prune_rate)
                logging.info(f"Pruned {name} with rate {self.prune_rate}")
        return self.model

    def save_checkpoint(self, checkpoint, filename='checkpoint.pth.tar'):
        # Save the model checkpoint to a file
        torch.save(checkpoint, filename)
        logging.info(f'Checkpoint saved to {filename}')

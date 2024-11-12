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

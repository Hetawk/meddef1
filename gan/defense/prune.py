import torch
import torch.nn.utils.prune as prune
import logging

class Pruner:
    def __init__(self, model, prune_rate):
        self.model = model
        self.prune_rate = prune_rate

    def unstructured_prune(self):
        """
        Apply L1 unstructured pruning to all Conv2d and Linear layers.
        This zeros out weights but keeps the parameter size unchanged.
        """
        # Track pruning statistics for logging
        pruned_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # Get parameter count before pruning
                param_size = module.weight.numel()
                total_params += param_size
                
                # Apply pruning - this creates a 'weight_mask' attribute
                prune.l1_unstructured(module, name='weight', amount=self.prune_rate)
                
                # Calculate how many parameters were pruned (zeros)
                pruned_count = param_size - torch.sum(module.weight_mask).item()
                pruned_params += pruned_count
                
                logging.info(f"Pruned {name}: {pruned_count}/{param_size} parameters ({pruned_count/param_size*100:.2f}%)")
        
        # Log overall pruning statistics
        if total_params > 0:
            overall_sparsity = pruned_params / total_params
            logging.info(f"Overall model sparsity: {overall_sparsity*100:.2f}%")
            logging.info(f"Total parameters: {total_params}, Pruned (zero) parameters: {pruned_params}, "
                       f"Remaining non-zero parameters: {total_params - pruned_params}")
        
        return self.model

    def save_checkpoint(self, checkpoint, filename='checkpoint.pth.tar'):
        torch.save(checkpoint, filename)
        logging.info(f'Checkpoint saved to {filename}')
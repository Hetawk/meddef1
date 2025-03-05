import torch
import gc
import logging
import numpy as np
import time
from typing import Optional, Dict, List, Callable, Tuple, Union, Any
from torch.cuda.amp import GradScaler, autocast
import math
import os
from datetime import datetime
from tqdm import tqdm

class MemoryEfficientTraining:
    """
    A utility class for memory-efficient training of deep learning models.
    Implements techniques like automatic batch size detection, gradient accumulation,
    and memory-optimized training workflows.
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable,
                 device: torch.device,
                 initial_batch_size: int = 32,
                 max_batch_size: int = 512,
                 find_batch_size: bool = True,
                 fp16: bool = True,
                 task_name: str = "normal_training",
                 dataset_name: str = "dataset",
                 model_name: str = "model",
                 epochs: int = 100,
                 lr: float = 0.001,
                 train_batch: int = 32,
                 adversarial: bool = False,
                 warn_grad_accum: int = 128):  # Add parameter to control warnings
        """
        Initialize the memory-efficient training utility.
        
        Args:
            model: The PyTorch model to train
            optimizer: The optimizer to use
            criterion: Loss function
            device: Device to train on
            initial_batch_size: Starting batch size for auto-detection
            max_batch_size: Maximum batch size to try
            find_batch_size: Whether to automatically find optimal batch size
            fp16: Whether to use mixed precision training
            task_name: Name of the training task
            dataset_name: Name of the dataset
            model_name: Name of the model
            epochs: Number of epochs
            lr: Learning rate
            train_batch: Training batch size
            adversarial: Whether to use adversarial training
            warn_grad_accum: Threshold for warning about large gradient accumulation steps (0 to disable)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.find_batch_size = find_batch_size
        self.fp16 = fp16
        self.scaler = GradScaler() if fp16 else None
        self.batch_size = initial_batch_size
        self.optimal_batch_size = initial_batch_size
        self.gradient_accumulation_steps = 1
        
        # Project-specific directory structure
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.train_batch = train_batch
        self.adversarial = adversarial  # Set from parameter, not hardcoded
        self.warn_grad_accum = warn_grad_accum  # Store the warning threshold
        
        # Setup autocast for mixed precision if enabled
        self.autocast_enabled = fp16

        # Set a more conservative initial batch size for large models
        n_params = sum(p.numel() for p in model.parameters()) / 1000000.0
        if n_params > 100:  # Very large model (>100M params)
            self.initial_batch_size = min(self.initial_batch_size, 8)
            logging.info(f"Large model detected ({n_params:.2f}M params), adjusting initial batch size to {self.initial_batch_size}")
        
        self.batch_size = self.initial_batch_size
        self.optimal_batch_size = self.initial_batch_size
        self.smallest_successful_batch = 1  # Fallback to absolute minimum

    def find_optimal_batch_size(self, dataloader: torch.utils.data.DataLoader) -> int:
        """
        Find the optimal batch size that fits in GPU memory.
        
        Args:
            dataloader: DataLoader to use for testing batch sizes
            
        Returns:
            The optimal batch size
        """
        logging.info(f"Finding optimal batch size (starting from {self.initial_batch_size}, max {self.max_batch_size})...")
        
        # Start with a small batch size and gradually increase
        test_batch_size = self.initial_batch_size
        largest_working_batch = 1  # Start with minimum as fallback
        
        # Save original model state
        original_state = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in self.optimizer.state_dict().items()}
        }
        
        self.model.train()
        
        try:
            # First try with very small batch size to establish a baseline
            for tiny_batch in [1, 2, 4]:
                if tiny_batch <= self.initial_batch_size:
                    self._clear_gpu_memory()
                    logging.info(f"Testing minimal batch size: {tiny_batch}")
                    success = self._test_batch_size(dataloader, tiny_batch)
                    if success:
                        largest_working_batch = tiny_batch
                        logging.info(f"✓ Batch size {tiny_batch} works as baseline")
                        if tiny_batch == 4:  # If we can do batch=4, we can try larger sizes
                            break
            
            # Now try to find the largest batch size that works
            if largest_working_batch > 0:
                test_batch_size = max(largest_working_batch * 2, self.initial_batch_size)
                while test_batch_size <= self.max_batch_size:
                    # Clear memory aggressively
                    self._clear_gpu_memory()
                    
                    # Try running a forward and backward pass with this batch size
                    logging.info(f"Testing batch size: {test_batch_size}")
                    success = self._test_batch_size(dataloader, test_batch_size)
                    
                    if success:
                        largest_working_batch = test_batch_size
                        logging.info(f"✓ Batch size {test_batch_size} works, trying larger batch size...")
                        test_batch_size *= 2
                        if test_batch_size > self.max_batch_size:
                            break
                    else:
                        # Try a smaller increment if binary search gets too coarse
                        if test_batch_size > largest_working_batch * 1.5 and test_batch_size // 1.5 > largest_working_batch:
                            test_batch_size = int(test_batch_size // 1.5)
                            logging.info(f"Trying intermediate batch size: {test_batch_size}")
                            success = self._test_batch_size(dataloader, test_batch_size)
                            if success:
                                largest_working_batch = test_batch_size
                        
                        logging.info(f"✗ Batch size {test_batch_size} caused OOM, optimal batch size: {largest_working_batch}")
                        break
                        
        except Exception as e:
            logging.error(f"Error during batch size finding: {str(e)}")
        finally:
            # Restore original model state
            self._clear_gpu_memory()
            self.model.load_state_dict(original_state['model'])
            self.optimizer.load_state_dict(original_state['optimizer'])
            
        # Set a safety margin by reducing the batch size slightly
        safety_factor = 0.8  # 80% of largest working batch for safety with large models
        optimal_batch_size = max(1, int(largest_working_batch * safety_factor))
        logging.info(f"Found optimal batch size: {optimal_batch_size} (safety factor: {safety_factor})")
        self.optimal_batch_size = optimal_batch_size
        self.smallest_successful_batch = max(1, largest_working_batch // 4)
        return optimal_batch_size
        
    def _test_batch_size(self, dataloader: torch.utils.data.DataLoader, batch_size: int) -> bool:
        """Test if a specific batch size works without OOM errors"""
        try:
            # Get a sample batch
            for inputs, targets in dataloader:
                if isinstance(inputs, torch.Tensor):
                    # Simulate a larger batch by repeating the data
                    if inputs.size(0) < batch_size:
                        repeat_factor = math.ceil(batch_size / inputs.size(0))
                        inputs = inputs.repeat(repeat_factor, 1, 1, 1)[:batch_size]
                        targets = targets.repeat(repeat_factor)[:batch_size]
                    else:
                        # Just take the first batch_size elements
                        inputs = inputs[:batch_size]
                        targets = targets[:batch_size]
                    
                    # Free memory from CPU tensors explicitly
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Move to device with non_blocking
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    # Ensure gradients are cleared
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass with mixed precision
                    with autocast(enabled=self.autocast_enabled):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    if self.fp16:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Clear memory of tensors
                    del inputs
                    del targets
                    del outputs
                    del loss
                
                # Test passed - do a second test to verify stability
                torch.cuda.empty_cache()
                gc.collect()
                
                # Try a simple forward pass to confirm we didn't just get lucky
                inputs = next(iter(dataloader))[0][:batch_size]
                inputs = inputs.to(self.device, non_blocking=True)
                with torch.no_grad(), autocast(enabled=self.autocast_enabled):
                    _ = self.model(inputs)
                del inputs
                
                # If we got here, it worked with this batch size
                return True
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "Unable to find a valid cuDNN algorithm" in str(e):
                return False
            else:
                logging.error(f"Error during batch size testing: {str(e)}")
                return False
        except Exception as e:
            logging.error(f"Unexpected error during batch size testing: {str(e)}")
            return False
            
    def _clear_gpu_memory(self):
        """Clear GPU memory cache more aggressively"""
        # Empty cache first
        torch.cuda.empty_cache()
        gc.collect()
        
        # Force garbage collection
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:
                pass
                
            # Report memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                cached = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                logging.debug(f"GPU memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
                
            # If high memory usage, try a more aggressive cleanup
            if allocated > 1000:  # If more than 1GB is allocated
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj):
                            obj.detach().cpu()
                    except:
                        pass
                torch.cuda.empty_cache()
                gc.collect()
    
    def get_effective_batch_size(self, target_batch_size: int) -> Tuple[int, int]:
        """
        Calculate actual batch size and gradient accumulation steps to 
        achieve the target batch size.
        
        Args:
            target_batch_size: Desired effective batch size
            
        Returns:
            Tuple of (actual_batch_size, gradient_accumulation_steps)
        """
        if target_batch_size <= self.optimal_batch_size:
            return target_batch_size, 1
            
        # Calculate steps needed for gradient accumulation to reach target
        actual_batch_size = self.optimal_batch_size
        grad_steps = math.ceil(target_batch_size / actual_batch_size)
        effective_batch = actual_batch_size * grad_steps
        
        logging.info(f"Using batch size {actual_batch_size} with {grad_steps} gradient accumulation steps " 
                    f"(effective batch size: {effective_batch})")
        
        # Only show warning if warn_grad_accum is enabled and steps exceed threshold
        if self.warn_grad_accum > 0 and grad_steps > self.warn_grad_accum:
            logging.warning(f"Using {grad_steps} gradient accumulation steps may affect convergence. "
                            f"Consider using a distributed training setup for very large batches.")
        
        return actual_batch_size, grad_steps
        
    def create_efficient_dataloader(self, 
                                   dataset: torch.utils.data.Dataset, 
                                   batch_size: int,
                                   shuffle: bool = True,
                                   num_workers: int = 4,
                                   pin_memory: bool = True) -> torch.utils.data.DataLoader:
        """
        Create a memory-efficient DataLoader with optimal settings.
        
        Args:
            dataset: The dataset to load
            batch_size: Batch size to use
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to use pinned memory
            
        Returns:
            Configured DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   epoch: int,
                   total_epochs: int,
                   grad_accum_steps: int = 1,
                   max_grad_norm: float = 1.0,
                   adversarial_trainer=None) -> dict:
        """
        Train the model for one epoch with memory-efficient settings.
        
        Args:
            dataloader: DataLoader providing the training data
            epoch: Current epoch number
            total_epochs: Total number of epochs
            grad_accum_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            adversarial_trainer: Optional adversarial training component
            
        Returns:
            Dictionary of training metrics
        """
        # Set adversarial flag if trainer is provided and log information about adversarial training
        if adversarial_trainer is not None:
            self.adversarial = True
            # Use safer attribute access with getattr
            attack_type = getattr(adversarial_trainer, 'attack_type', 
                                 getattr(adversarial_trainer, 'attack_name', 'unknown'))
            epsilon = getattr(adversarial_trainer, 'epsilon', 0.2)
            adv_weight = getattr(adversarial_trainer, 'adv_weight', 0.5)
            logging.info(f"Epoch {epoch+1}: Using adversarial training with {attack_type}, epsilon={epsilon}, weight={adv_weight}")
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_loss = 0.0
        adv_correct = 0
        adv_loss_sum = 0.0
        start_time = time.time()
        
        # Track additional metrics
        epoch_true_labels = []
        epoch_predictions = []
        
        # Reset gradients at the start of the epoch
        self.optimizer.zero_grad(set_to_none=True)
        
        # Set up progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch")
        
        # Determine a safe chunk size based on experience so far
        chunk_size = max(1, self.optimal_batch_size // 2)  # Make chunks even smaller for safety
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            try:
                # Move data to device one chunk at a time
                if isinstance(data, torch.Tensor) and data.shape[0] > chunk_size:
                    # Split into smaller chunks for processing
                    data_chunks = []
                    target_chunks = []
                    
                    # Create chunks of safe size
                    num_chunks = math.ceil(data.shape[0] / chunk_size)
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, data.shape[0])
                        data_chunks.append(data[start_idx:end_idx])
                        target_chunks.append(target[start_idx:end_idx])
                    
                    # Process each chunk with aggressive memory clearing between chunks
                    chunk_results = []
                    for chunk_idx, (data_chunk, target_chunk) in enumerate(zip(data_chunks, target_chunks)):
                        # Clear memory before processing each chunk
                        if chunk_idx > 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        # Process this chunk
                        result = self._process_single_chunk(
                            data_chunk, target_chunk, grad_accum_steps, batch_idx, 
                            chunk_idx, len(data_chunks), adversarial_trainer
                        )
                        chunk_results.append(result)
                    
                    # Combine results
                    batch_metrics = {
                        k: sum(result.get(k, 0) for result in chunk_results)
                        for k in ['loss', 'correct', 'total', 'adv_loss', 'adv_correct']
                    }
                else:
                    # Process the entire batch if it's small enough
                    batch_metrics = self._process_batch(data, target, grad_accum_steps, batch_idx, adversarial_trainer)
                
                # Update metrics
                total_loss += batch_metrics['loss']
                correct += batch_metrics['correct']
                total += batch_metrics['total']
                
                # Get predictions in a memory-efficient way
                if isinstance(data, torch.Tensor) and data.shape[0] <= chunk_size:
                    with torch.no_grad(), autocast(enabled=self.autocast_enabled):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        pred = self.model(data).argmax(dim=1)
                        epoch_true_labels.extend(target.cpu().numpy())
                        epoch_predictions.extend(pred.cpu().numpy())
                        del data, target, pred
                
                if adversarial_trainer:
                    adv_loss_sum += batch_metrics.get('adv_loss', 0)
                    adv_correct += batch_metrics.get('adv_correct', 0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{batch_metrics['loss']:.4f}",
                    'acc': f"{correct/total:.4f}" if total > 0 else "N/A"
                })
                
                # Optimizer step after accumulation
                if (batch_idx + 1) % grad_accum_steps == 0:
                    self._optimizer_step(max_grad_norm)
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "Unable to find a valid cuDNN algorithm" in str(e):
                    logging.error(f"OOM in batch {batch_idx}: {str(e)}")
                    
                    # Save checkpoint to resume later
                    self._save_checkpoint(epoch, batch_idx)
                    
                    # Clear memory and try to continue with smaller chunks
                    self._clear_gpu_memory()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Reduce chunk size for future batches
                    chunk_size = max(1, chunk_size // 2)
                    self.optimal_batch_size = max(1, self.optimal_batch_size // 2)
                    logging.info(f"Reduced chunk size to {chunk_size} and optimal batch size to {self.optimal_batch_size}")
                    
                    continue
                else:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
            except Exception as e:
                logging.error(f"Unexpected error in batch {batch_idx}: {str(e)}")
                self.optimizer.zero_grad(set_to_none=True)
                continue
        
        # Do final optimization step if needed
        if len(dataloader) % grad_accum_steps != 0:
            self._optimizer_step(max_grad_norm)
            
        # Calculate epoch statistics
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        # Calculate adversarial statistics if applicable
        adv_accuracy = 0
        avg_adv_loss = 0
        if adversarial_trainer:
            avg_adv_loss = adv_loss_sum / len(dataloader) if len(dataloader) > 0 else 0
            adv_accuracy = adv_correct / total if total > 0 else 0
            
        # Always log both if adversarial flag is set (even from initialization)
        if self.adversarial or adversarial_trainer:
            logging.info(
                f'Epoch {epoch+1} Training - Clean: Loss={avg_loss:.4f}, Acc={accuracy:.4f} | '
                f'Adversarial: Loss={avg_adv_loss:.4f}, Acc={adv_accuracy:.4f} | '
            )
        else:
            logging.info(
                f'Epoch {epoch+1} Training - Loss={avg_loss:.4f}, Acc={accuracy:.4f} | '
            )
            
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'adv_loss': avg_adv_loss,
            'adv_accuracy': adv_accuracy,
            'correct': correct,
            'total': total,
            'true_labels': epoch_true_labels,
            'predictions': epoch_predictions
        }
            
    def _process_batch(self, data, target, grad_accum_steps, batch_idx, adversarial_trainer=None):
        """Process a single batch with optional adversarial training"""
        metrics = {'loss': 0, 'correct': 0, 'total': 0, 'adv_loss': 0, 'adv_correct': 0}
        
        with autocast(enabled=self.autocast_enabled):
            # Forward pass
            output = self.model(data)
            clean_loss = self.criterion(output, target)
            loss = clean_loss  # Start with clean loss
            
            # Calculate adversarial loss if enabled
            if adversarial_trainer:
                try:
                    # Generate adversarial examples with error handling
                    with torch.cuda.amp.autocast(enabled=False):
                        epsilon = getattr(adversarial_trainer, 'epsilon', 0.2)
                        if hasattr(adversarial_trainer.attack, 'generate'):
                            adv_data = adversarial_trainer.attack.generate(data, target, epsilon)
                        else:
                            _, adv_data, _ = adversarial_trainer.attack.attack(data, target)
                    
                    # Calculate loss on adversarial examples
                    with autocast(enabled=self.autocast_enabled):
                        adv_output = self.model(adv_data)
                        adv_loss = self.criterion(adv_output, target)
                    
                    # Combine losses using adversarial weight
                    w = float(getattr(adversarial_trainer, 'adv_weight', 0.5))
                    loss = (1 - w) * clean_loss + w * adv_loss
                    
                    # Track adversarial metrics
                    adv_pred = adv_output.argmax(dim=1, keepdim=True)
                    metrics['adv_correct'] = adv_pred.eq(target.view_as(adv_pred)).sum().item()
                    metrics['adv_loss'] = adv_loss.item()
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.warning(f"OOM during adversarial example generation, skipping adversarial part")
                        # Continue with just the clean loss
                    else:
                        logging.error(f"Error in adversarial training: {str(e)}")
                        # Continue with just the clean loss
            
            # Scale loss by accumulation steps
            scaled_loss = loss / grad_accum_steps
        
        # Backward pass
        if self.fp16:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track metrics - store unscaled loss for reporting
        with torch.no_grad():
            pred = output.argmax(dim=1, keepdim=True)
            metrics['correct'] = pred.eq(target.view_as(pred)).sum().item()
            metrics['total'] = target.size(0)
            metrics['loss'] = clean_loss.item()  # Store the clean loss for consistent reporting
        
        return metrics
    
    def _optimizer_step(self, max_grad_norm):
        """Perform optimization step with gradient clipping"""
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
    
    def validate(self, dataloader, adversarial_trainer=None):
        """
        Validate model on validation dataset with memory optimizations
        
        Args:
            dataloader: Validation dataloader
            adversarial_trainer: Optional adversarial training component
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0
        adv_val_loss = 0
        correct = 0
        adv_correct = 0
        total = 0
        
        try:
            with torch.no_grad():
                for data, target in tqdm(dataloader, desc="Validating", unit="batch"):
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device, non_blocking=True)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.device, non_blocking=True)
                    
                    # Process in smaller chunks if needed
                    if data.shape[0] > self.optimal_batch_size:
                        data_chunks = data.chunk(math.ceil(data.shape[0] / self.optimal_batch_size))
                        target_chunks = target.chunk(math.ceil(target.shape[0] / self.optimal_batch_size))
                        
                        chunk_results = []
                        for data_chunk, target_chunk in zip(data_chunks, target_chunks):
                            chunk_results.append(
                                self._process_validation_batch(data_chunk, target_chunk, adversarial_trainer)
                            )
                        
                        # Combine results
                        batch_metrics = {
                            k: sum(result.get(k, 0) for result in chunk_results)
                            for k in ['loss', 'correct', 'total', 'adv_loss', 'adv_correct']
                        }
                    else:
                        batch_metrics = self._process_validation_batch(data, target, adversarial_trainer)
                    
                    # Update metrics
                    val_loss += batch_metrics['loss']
                    correct += batch_metrics['correct']
                    total += batch_metrics['total']
                    if adversarial_trainer:
                        adv_val_loss += batch_metrics.get('adv_loss', 0)
                        adv_correct += batch_metrics.get('adv_correct', 0)
        except Exception as e:
            logging.error(f"Error during validation: {str(e)}")
            return {'val_loss': float('inf'), 'val_accuracy': 0.0}
        finally:
            self.model.train()
            
        val_loss /= len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        result = {'val_loss': val_loss, 'val_accuracy': accuracy}
        
        # Add adversarial metrics if applicable
        if adversarial_trainer:
            adv_val_loss /= len(dataloader)
            adv_accuracy = adv_correct / total if total > 0 else 0
            result.update({
                'adv_val_loss': adv_val_loss,
                'adv_val_accuracy': adv_accuracy
            })
            logging.info(
                f'Validation - Clean: Loss={val_loss:.4f}, Acc={accuracy:.4f} | '
                f'Adversarial: Loss={adv_val_loss:.4f}, Acc={adv_accuracy:.4f}'
            )
        else:
            logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
            
        return result
    
    def _process_validation_batch(self, data, target, adversarial_trainer=None):
        """Process a single validation batch"""
        metrics = {'loss': 0, 'correct': 0, 'total': 0, 'adv_loss': 0, 'adv_correct': 0}
        
        with autocast(enabled=self.autocast_enabled):
            output = self.model(data)
            loss = self.criterion(output, target)
        
        metrics['loss'] = loss.item()
        
        # Calculate adversarial metrics if enabled
        if adversarial_trainer:
            try:
                with torch.enable_grad():
                    epsilon = getattr(adversarial_trainer, 'epsilon', 0.2)
                    if hasattr(adversarial_trainer.attack, 'generate'):
                        adv_data = adversarial_trainer.attack.generate(data, target, epsilon)
                    else:
                        _, adv_data, _ = adversarial_trainer.attack.attack(data, target)
                
                with autocast(enabled=self.autocast_enabled):
                    adv_output = self.model(adv_data)
                    adv_loss = self.criterion(adv_output, target)
                
                metrics['adv_loss'] = adv_loss.item()
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                metrics['adv_correct'] = adv_pred.eq(target.view_as(adv_pred)).sum().item()
            except Exception as e:
                logging.warning(f"Error in adversarial validation: {str(e)}")
        
        pred = output.argmax(dim=1, keepdim=True)
        metrics['correct'] = pred.eq(target.view_as(pred)).sum().item()
        metrics['total'] = target.size(0)
        
        return metrics
    
    def _save_checkpoint(self, epoch, batch_idx):
        """Save checkpoint for resuming training"""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.fp16 else None,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d")
        # Match the pattern used in train.py
        filename = f"best_{self.model_name}_{self.dataset_name}_epochs{self.epochs}_lr{self.lr}_batch{self.train_batch}_{timestamp}.pth"
        
        # Create directory structure matching train.py's save_model pattern
        if hasattr(self, 'adversarial') and self.adversarial:
            checkpoint_dir = os.path.join('out', self.task_name, self.dataset_name, self.model_name, 'adv', 'save_model')
        else:
            checkpoint_dir = os.path.join('out', self.task_name, self.dataset_name, self.model_name, 'save_model')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights in the same format as train.py
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(self.model.state_dict(), checkpoint_path)  
        logging.info(f"Saved model checkpoint to {checkpoint_path}")
        
        # Additionally save full checkpoint for resuming training
        resume_filename = f"resume_epoch{epoch}_batch{batch_idx}_epochs{self.epochs}_lr{self.lr}_batch{self.train_batch}_{timestamp}.pth"
        resume_path = os.path.join(checkpoint_dir, resume_filename)
        torch.save(checkpoint, resume_path)
        logging.info(f"Saved full training state to {resume_path}")
        
    def load_checkpoint(self, checkpoint_path=None):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Optional path to checkpoint file.
                If None, will try to find latest checkpoint.
                
        Returns:
            Tuple of (success, epoch, batch_idx)
        """
        if checkpoint_path is None:
            # First look for resume checkpoints
            checkpoint_dir = os.path.join('out', self.task_name, self.dataset_name, self.model_name, 'save_model')
            if hasattr(self, 'adversarial') and self.adversarial:
                checkpoint_dir = os.path.join('out', self.task_name, self.dataset_name, self.model_name, 'adv', 'save_model')
                
            if not os.path.exists(checkpoint_dir):
                logging.warning(f"No checkpoint directory found at {checkpoint_dir}")
                return False, 0, 0
                
            # First try to find resume files
            resume_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('resume_')]
            
            if resume_files:
                # Sort by modification time (newest first)
                resume_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                checkpoint_path = os.path.join(checkpoint_dir, resume_files[0])
                logging.info(f"Found latest resume checkpoint: {checkpoint_path}")
                
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if self.fp16 and 'scaler' in checkpoint and checkpoint['scaler']:
                        self.scaler.load_state_dict(checkpoint['scaler'])
                    epoch = checkpoint.get('epoch', 0)
                    batch_idx = checkpoint.get('batch_idx', 0)
                    logging.info(f"Resumed training from epoch {epoch}, batch {batch_idx}")
                    return True, epoch, batch_idx
                except Exception as e:
                    logging.error(f"Error loading resume checkpoint: {str(e)}")
                    return False, 0, 0
            
            # Fall back to regular model checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"best_{self.model_name}")]
            if not checkpoints:
                logging.warning(f"No checkpoints found in {checkpoint_dir}")
                return False, 0, 0
            
            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
            logging.info(f"Found latest model checkpoint: {checkpoint_path}")
                
            # This is just the model state dict, not the full training state
            try:
                # Match the loading pattern in ModelLoader
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # Handle DataParallel saved models
                new_state_dict = {}
                for k, v in checkpoint.items():
                    new_key = k.replace("module.", "") if k.startswith("module.") else k
                    new_state_dict[new_key] = v
                    
                self.model.load_state_dict(new_state_dict)
                logging.info(f"Loaded model weights from {checkpoint_path}")
                # Note: We can't resume exact training state, just the model weights
                return True, 0, 0
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
                return False, 0, 0
        else:
            if not os.path.exists(checkpoint_path):
                logging.error(f"Checkpoint not found: {checkpoint_path}")
                return False, 0, 0
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.fp16 and 'scaler' in checkpoint and checkpoint['scaler']:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                epoch = checkpoint.get('epoch', 0)
                batch_idx = checkpoint.get('batch_idx', 0)
                logging.info(f"Loaded checkpoint from epoch {epoch}, batch {batch_idx}")
                return True, epoch, batch_idx
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
                return False, 0, 0

    def _process_single_chunk(self, data, target, grad_accum_steps, batch_idx, chunk_idx, total_chunks, adversarial_trainer=None):
        """Process a single data chunk with extra memory management"""
        try:
            # Move this chunk to device
            if isinstance(data, torch.Tensor):
                data = data.to(self.device, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device, non_blocking=True)
            
            # Process batch normally
            result = self._process_batch(data, target, grad_accum_steps * total_chunks, batch_idx, adversarial_trainer)
            
            # Immediately clear these tensors from GPU
            del data
            del target
            
            return result
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "Unable to find a valid cuDNN algorithm" in str(e):
                # Try even smaller chunks in a recursive manner
                if data.shape[0] > 1:
                    logging.warning(f"OOM in chunk {chunk_idx}, trying with smaller chunks")
                    
                    # Split this chunk in half
                    mid = data.shape[0] // 2
                    data1, data2 = data[:mid], data[mid:]
                    target1, target2 = target[:mid], target[mid:]
                    
                    # Clear the original tensors
                    del data
                    del target
                    torch.cuda.empty_cache()
                    
                    # Process each half
                    result1 = self._process_single_chunk(
                        data1, target1, grad_accum_steps, batch_idx, 
                        f"{chunk_idx}.1", total_chunks * 2, adversarial_trainer
                    )
                    
                    # Clear memory from first half
                    del data1
                    del target1
                    torch.cuda.empty_cache()
                    
                    result2 = self._process_single_chunk(
                        data2, target2, grad_accum_steps, batch_idx, 
                        f"{chunk_idx}.2", total_chunks * 2, adversarial_trainer
                    )
                    
                    # Combine results
                    combined_result = {
                        k: result1.get(k, 0) + result2.get(k, 0)
                        for k in ['loss', 'correct', 'total', 'adv_loss', 'adv_correct']
                    }
                    
                    return combined_result
                else:
                    # Cannot split further, single sample is too big
                    logging.error("Unable to process even a single sample due to memory limitations")
                    return {'loss': 0, 'correct': 0, 'total': 0, 'adv_loss': 0, 'adv_correct': 0}
            else:
                logging.error(f"Error in chunk {chunk_idx}: {str(e)}")
                return {'loss': 0, 'correct': 0, 'total': 0, 'adv_loss': 0, 'adv_correct': 0}

import torch
import gc
import logging
import numpy as np
from typing import Optional, Dict, List
from collections import OrderedDict

class MemoryEfficientModel:
    def __init__(self, model_builder, device, fp16=False, min_chunk_size=1):
        self.model_builder = model_builder
        self.device = device
        self.fp16 = fp16
        self.min_chunk_size = min_chunk_size * 1024 * 1024  # Convert MB to bytes
        self.param_groups: Dict[str, torch.Tensor] = {}

    def load_model(self, **kwargs):
        model = self.model_builder(**kwargs)
        try:
            if self.fp16:
                model = model.half()

            # Enable gradient checkpointing if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

            # Keep model on CPU initially
            model = model.cpu()
            
            # Get model structure by layer types
            layer_groups = self._group_parameters_by_layer(model)
            
            # Try loading each group separately
            success = False
            for group_name, params in layer_groups.items():
                try:
                    self._clear_gpu_memory()
                    logging.info(f"Loading {group_name} parameters...")
                    
                    # Load parameters for this group in micro-batches
                    self._load_parameter_group(params, model)
                    success = True
                except RuntimeError as e:
                    logging.warning(f"Failed to load {group_name}: {str(e)}")
                    self._clear_gpu_memory()
                    continue

            if not success:
                raise RuntimeError("Failed to load any parameter groups")

            # Final attempt to assemble model with minimal memory usage
            self._clear_gpu_memory()
            model = self._assemble_model_minimal_memory(model)
            
            return model

        except Exception as e:
            self._clear_gpu_memory()
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _group_parameters_by_layer(self, model):
        groups = OrderedDict()
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                layer_type = module.__class__.__name__
                if layer_type not in groups:
                    groups[layer_type] = []
                for param_name, param in module.named_parameters():
                    full_name = f"{name}.{param_name}"
                    groups[layer_type].append((full_name, param))
        return groups

    def _load_parameter_group(self, params, model):
        total_size = sum(p[1].numel() * p[1].element_size() for p in params)
        num_chunks = max(1, total_size // self.min_chunk_size)
        chunk_size = total_size // num_chunks
        
        current_chunk = []
        current_size = 0
        
        for name, param in params:
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > chunk_size and current_chunk:
                self._process_chunk(current_chunk, model)
                current_chunk = []
                current_size = 0
            
            current_chunk.append((name, param))
            current_size += param_size
        
        if current_chunk:
            self._process_chunk(current_chunk, model)

    def _process_chunk(self, chunk, model):
        total_chunk_params = len(chunk)
        total_chunk_size = sum(param.numel() * param.element_size() for _, param in chunk)
        # logging.info(f"Processing chunk with {total_chunk_params} parameters ({total_chunk_size} bytes)...")
        for name, param in chunk:
            try:
                # Pin memory and transfer in smallest possible units
                param_cpu = param.detach().cpu()
                if self.fp16:
                    param_cpu = param_cpu.half()
                param_cpu = param_cpu.pin_memory()
                
                # Transfer to GPU with minimal additional memory
                param_gpu = param_cpu.to(self.device, non_blocking=True)
                
                # Set parameter in model
                *parent_path, param_name = name.split('.')
                current = model
                for part in parent_path:
                    current = getattr(current, part)
                setattr(current, param_name, torch.nn.Parameter(param_gpu))
                
                del param_cpu
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Failed to process parameter '{name}' in chunk (chunk size: {total_chunk_size} bytes): {str(e)}")
                raise

    def _assemble_model_minimal_memory(self, model):
        try:
            # Move model structure (without parameters) to GPU
            model = model.to(self.device)
            
            # Verify all parameters are on correct device
            for name, param in model.named_parameters():
                if param.device != self.device:
                    param.data = param.data.to(self.device, non_blocking=True)
            
            return model
        except Exception as e:
            logging.error(f"Failed to assemble model: {str(e)}")
            raise

    def _clear_gpu_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:
                pass

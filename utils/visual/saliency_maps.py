import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from typing import List, Optional, Dict, Tuple


class SaliencyMapGenerator:
    """Generate saliency maps for model interpretability"""
    
    def __init__(self, device=None):
        """
        Initialize the saliency map generator
        
        Args:
            device: torch device to use (defaults to GPU if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def _normalize_saliency(self, saliency: torch.Tensor) -> np.ndarray:
        """Normalize saliency map for visualization"""
        saliency = saliency.abs().cpu().numpy()
        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency
        
    def generate_vanilla_saliency(self, model, image, target_class=None) -> torch.Tensor:
        """
        Generate vanilla saliency map (gradients of output w.r.t. input)
        
        Args:
            model: PyTorch model
            image: Input image tensor [C, H, W]
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            saliency_map: Saliency map tensor
        """
        # Set model to eval mode
        model.eval()
        
        # Convert to tensor and add batch dimension
        if isinstance(image, np.ndarray):
            image = self.transform(image).to(self.device)
        elif isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0).to(self.device)
            
        # Enable gradients for input
        image.requires_grad_()
        
        # Forward pass
        output = model(image)
        
        # If target class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Zero gradients
        if hasattr(model, 'zero_grad'):
            model.zero_grad()
        else:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
                    
        # Compute gradients w.r.t input
        target = output[0, target_class]
        target.backward()
        
        # Get gradients
        saliency_map = image.grad[0].abs()
        
        # If 3-channel, take mean across channels
        if saliency_map.shape[0] == 3:
            saliency_map = saliency_map.mean(dim=0, keepdim=True)
            
        return saliency_map
        
    def generate_guided_backprop(self, model, image, target_class=None) -> torch.Tensor:
        """
        Generate guided backpropagation saliency map
        
        Args:
            model: PyTorch model
            image: Input image tensor [C, H, W]
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            saliency_map: Saliency map tensor
        """
        # Register hooks to modify gradients during backward pass
        handles = []
        
        def backward_hook(module, grad_in, grad_out):
            # Clip negative gradients to zero for guided backprop
            if isinstance(grad_in, tuple):
                revised_grad_in = []
                for g in grad_in:
                    if g is not None:
                        g = g.clamp(min=0)
                    revised_grad_in.append(g)
                return tuple(revised_grad_in)
            else:
                return grad_in.clamp(min=0)
        
        # Register hooks for all ReLU layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_backward_hook(backward_hook)
                handles.append(handle)
        
        # Generate saliency map
        saliency_map = self.generate_vanilla_saliency(model, image, target_class)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return saliency_map
    
    def generate_integrated_gradients(self, model, image, target_class=None, steps=50) -> torch.Tensor:
        """
        Generate integrated gradients saliency map
        
        Args:
            model: PyTorch model
            image: Input image tensor [C, H, W]
            target_class: Target class index (if None, uses predicted class)
            steps: Number of steps for integration
            
        Returns:
            saliency_map: Integrated gradients saliency map tensor
        """
        # Ensure input is a tensor
        if isinstance(image, np.ndarray):
            image = self.transform(image).to(self.device)
        elif isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0).to(self.device)
            
        # Create baseline (black image)
        baseline = torch.zeros_like(image, device=self.device)
        
        # Forward pass to determine target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = model(image)
                target_class = output.argmax(dim=1).item()
        
        # Generate interpolated images
        alphas = torch.linspace(0, 1, steps, device=self.device)
        
        # Create steps interpolated images
        interpolated_images = []
        for alpha in alphas:
            interpolated_images.append(alpha * image + (1 - alpha) * baseline)
        
        # Stack interpolated images
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
        # Get gradients for all interpolated images
        integrated_gradients = torch.zeros_like(image, device=self.device)
        batch_size = 10  # Process in smaller batches to avoid memory issues
        
        for i in range(0, steps, batch_size):
            end = min(i + batch_size, steps)
            batch = interpolated_images[i:end]
            
            batch.requires_grad_()
            output = model(batch)
            
            # Create a one-hot encoding for the target class
            target = torch.zeros_like(output)
            target[:, target_class] = 1
            
            # Zero gradients
            model.zero_grad()
            
            # Backward pass
            output.backward(gradient=target)
            
            # Sum gradients
            gradient = batch.grad.clone()
            integrated_gradients += gradient.sum(dim=0, keepdim=True)
        
        # Scale by 1/steps and multiply by input - baseline
        integrated_gradients = integrated_gradients * (image - baseline) / steps
        
        # Take mean across channels if 3-channel
        if integrated_gradients.shape[1] == 3:
            integrated_gradients = integrated_gradients.mean(dim=1, keepdim=True)
            
        return integrated_gradients[0]
        
    def visualize_saliency(self, image, saliency_map, save_path=None, title=None, cmap='jet'):
        """
        Visualize saliency map overlaid on original image
        
        Args:
            image: Original image (numpy array or tensor)
            saliency_map: Saliency map tensor
            save_path: Path to save the visualization
            title: Title for the plot
            cmap: Colormap for saliency map
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert image to numpy and denormalize if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # batch dimension
                image = image[0]
            image = image.cpu().numpy().transpose(1, 2, 0)
            
        # Handle single-channel images
        if image.shape[-1] == 1:
            image = image.squeeze()
            
        # Normalize image if needed
        if image.max() > 1:
            image = image / 255.0
            
        # Convert saliency map to numpy
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = self._normalize_saliency(saliency_map)
        
        # If saliency is 3D, take the first channel or mean across channels
        if saliency_map.ndim == 3:
            if saliency_map.shape[0] == 3:  # RGB saliency
                saliency_map = saliency_map.mean(axis=0)
            else:  # Has a single channel dimension
                saliency_map = saliency_map.squeeze(0)
        
        # Create figure
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ax[0].imshow(image, cmap='gray')
        else:
            ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        # Plot saliency map
        saliency_img = ax[1].imshow(saliency_map, cmap=cmap)
        ax[1].set_title('Saliency Map')
        ax[1].axis('off')
        plt.colorbar(saliency_img, ax=ax[1], fraction=0.046, pad=0.04)
        
        # Plot overlay
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            # Convert grayscale to RGB for overlay
            rgb_img = np.stack([image, image, image], axis=2) if image.ndim == 2 else np.repeat(image, 3, axis=2)
        else:
            rgb_img = image
            
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay saliency on image
        overlay = 0.7 * rgb_img + 0.3 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        ax[2].imshow(overlay)
        ax[2].set_title('Overlay')
        ax[2].axis('off')
        
        # Set main title if provided
        if title:
            plt.suptitle(title, fontsize=15)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


def compare_saliency_maps(models: List, images: List,
                          targets: Optional[List] = None,
                          model_names: Optional[List[str]] = None,
                          save_dir: Optional[str] = None,
                          dataset_name: str = '',
                          method: str = 'vanilla',
                          rows_per_image: int = 1) -> List:
    """
    Compare saliency maps across multiple models for given images
    
    Args:
        models: List of models to compare
        images: List of images (numpy arrays or tensors)
        targets: Optional list of target classes for each image
        model_names: Optional list of model names
        save_dir: Directory to save visualizations
        dataset_name: Name of dataset (for saving)
        method: Saliency method ('vanilla', 'guided', 'integrated')
        rows_per_image: Number of model rows to show per image
        
    Returns:
        figures: List of matplotlib figures
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SaliencyMapGenerator(device)
    figures = []
    
    # Default model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
        
    # Default targets to None for each image
    if targets is None:
        targets = [None] * len(images)
        
    # Set method function
    if method == 'guided':
        method_fn = generator.generate_guided_backprop
    elif method == 'integrated':
        method_fn = generator.generate_integrated_gradients
    else:
        method_fn = generator.generate_vanilla_saliency
        
    # Process each image
    for img_idx, (image, target) in enumerate(zip(images, targets)):
        # Create plot with enough rows for all models
        n_rows = (len(models) + rows_per_image - 1) // rows_per_image
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        # Process each model
        for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
            row_idx = model_idx // rows_per_image
            
            # Move model to device
            model = model.to(device)
            model.eval()
            
            # Generate saliency map
            saliency_map = method_fn(model, image, target)
            
            # Convert image to numpy and denormalize if needed
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # batch dimension
                    image_np = image[0]
                else:
                    image_np = image
                image_np = image_np.cpu().numpy().transpose(1, 2, 0)
            else:
                image_np = image
                
            # Handle single-channel images
            if image_np.shape[-1] == 1:
                image_np = image_np.squeeze()
                
            # Normalize image if needed
            if image_np.max() > 1:
                image_np = image_np / 255.0
                
            # Normalize saliency map
            saliency_np = generator._normalize_saliency(saliency_map)
            
            # If saliency is 3D, take the first channel or mean across channels
            if saliency_np.ndim == 3:
                if saliency_np.shape[0] == 3:  # RGB saliency
                    saliency_np = saliency_np.mean(axis=0)
                else:  # Has a single channel dimension
                    saliency_np = saliency_np.squeeze(0)
            
            # Plot original image (only once per row)
            if model_idx % rows_per_image == 0:
                if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[-1] == 1):
                    axes[row_idx, 0].imshow(image_np, cmap='gray')
                else:
                    axes[row_idx, 0].imshow(image_np)
                axes[row_idx, 0].set_title('Original Image')
                axes[row_idx, 0].axis('off')
                
            # Plot saliency map
            saliency_img = axes[row_idx, 1].imshow(saliency_np, cmap='jet')
            axes[row_idx, 1].set_title(f'Saliency: {model_name}')
            axes[row_idx, 1].axis('off')
            plt.colorbar(saliency_img, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
            
            # Plot overlay
            if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[-1] == 1):
                # Convert grayscale to RGB for overlay
                rgb_img = np.stack([image_np, image_np, image_np], axis=2) if image_np.ndim == 2 else np.repeat(image_np, 3, axis=2)
            else:
                rgb_img = image_np
                
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * saliency_np), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # Overlay saliency on image
            overlay = 0.7 * rgb_img + 0.3 * heatmap
            overlay = np.clip(overlay, 0, 1)
            
            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title(f'Overlay: {model_name}')
            axes[row_idx, 2].axis('off')
        
        plt.suptitle(f"Saliency Maps Comparison - {method.title()} Method", fontsize=15)
        plt.tight_layout()
        
        # Save figure if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 
                                    f"saliency_comparison_{dataset_name}_img{img_idx}_{method}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        figures.append(fig)
        
    return figures


def compare_pruned_saliency(base_model, pruning_rates, image, 
                            target=None, device=None, method='vanilla',
                            save_dir=None, model_name='model', dataset_name=''):
    """
    Compare saliency maps for different pruning rates of the same model
    
    Args:
        base_model: Base unpruned model
        pruning_rates: List of pruning rates to compare
        image: Input image
        target: Optional target class
        device: Torch device
        method: Saliency method ('vanilla', 'guided', 'integrated')
        save_dir: Directory to save visualizations
        model_name: Name of base model
        dataset_name: Name of dataset
        
    Returns:
        fig: Matplotlib figure
    """
    from gan.defense.prune import Pruner
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SaliencyMapGenerator(device)
    
    # Set method function
    if method == 'guided':
        method_fn = generator.generate_guided_backprop
    elif method == 'integrated':
        method_fn = generator.generate_integrated_gradients
    else:
        method_fn = generator.generate_vanilla_saliency
    
    # Create models with different pruning rates
    models = [base_model]  # Start with unpruned model
    model_names = [f"{model_name} (Unpruned)"]
    
    for rate in pruning_rates:
        pruner = Pruner(base_model, rate)
        pruned_model = pruner.unstructured_prune()
        models.append(pruned_model)
        model_names.append(f"{model_name} (Pruned {rate*100:.0f}%)")
    
    # Compare saliency maps
    figures = compare_saliency_maps(
        models=models,
        images=[image],
        targets=[target],
        model_names=model_names,
        save_dir=save_dir,
        dataset_name=dataset_name,
        method=method
    )
    
    return figures[0]

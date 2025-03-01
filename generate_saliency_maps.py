#!/usr/bin/env python3

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Generate saliency maps for a model')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--arch', type=str, required=True, help='Model architecture')
    parser.add_argument('--depth', type=float, required=True, help='Model depth/variant')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image_paths', nargs='+', help='Paths to the images for saliency map generation')
    parser.add_argument('--output_dir', type=str, default='out/saliency_maps', help='Output directory for saliency maps')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0], help='GPU IDs to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (default: 4)')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin_memory')
    parser.add_argument('--combine', action='store_true', help='Combine multiple images into a single output')
    return parser.parse_args()

class SaliencyMapGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu")
        
        # Load dataset to get number of classes
        self.dataset_loader = DatasetLoader()
        _, _, test_loader = self.dataset_loader.load_data(
            dataset_name=args.data,
            batch_size={
                'train': args.batch_size,
                'val': args.batch_size,
                'test': args.batch_size
            },
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if hasattr(args, 'pin_memory') else False
        )
        
        # Get number of classes
        dataset = test_loader.dataset
        if hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            self.num_classes = len(dataset.class_to_idx)
        else:
            raise AttributeError("Dataset does not contain class information")
            
        # Get the preprocessing transform from the dataset if available
        if hasattr(dataset, 'transform'):
            self.transform = dataset.transform
        else:
            # Fallback to standard normalization transform
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Get the class names if available
        if hasattr(dataset, 'classes'):
            self.class_names = dataset.classes
        elif hasattr(dataset, 'class_to_idx'):
            self.class_names = list(dataset.class_to_idx.keys())
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]
        
        # Load model
        self.model_loader = ModelLoader(self.device, args.arch, pretrained=False)
        self.model = self.load_model()
        
        # Create output directory
        self.output_dir = os.path.join(args.output_dir, args.data, f"{args.arch}_{args.depth}")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self):
        # Get the model from the model loader
        models_and_names = self.model_loader.get_model(
            model_name=self.args.arch,
            depth=float(self.args.depth),
            input_channels=3,
            num_classes=self.num_classes
        )
        
        if not models_and_names:
            raise ValueError("No models returned from model loader")
        
        model, _ = models_and_names[0]
        model = model.to(self.device)
        
        # Load weights
        if os.path.exists(self.args.model_path):
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {self.args.model_path}")
        else:
            raise FileNotFoundError(f"Model path {self.args.model_path} does not exist")
        
        model.eval()
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess the image for model input"""
        # Load the image and save original
        original_image = Image.open(image_path).convert('RGB')
        
        # Apply transformations for model input
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        return input_tensor, original_image
    
    def compute_vanilla_saliency(self, input_tensor, target_class=None):
        """Compute vanilla gradient saliency map"""
        # Enable gradient calculation for input
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target class is not provided, use the predicted class
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get gradient with respect to input
        gradients = input_tensor.grad.data
        
        # Calculate saliency map (absolute value of gradients)
        saliency_map = torch.abs(gradients)
        saliency_map = torch.max(saliency_map, dim=1)[0].unsqueeze(1)
        
        # Normalize saliency map for visualization
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map
    
    def compute_gradcam(self, input_tensor, target_layer_name=None, target_class=None):
        """Compute Grad-CAM saliency map"""
        # Set architecture-specific target layers if not specified
        if target_layer_name is None:
            if 'densenet' in self.args.arch.lower():
                # For DenseNet models, use the final dense block or transition layer
                target_layer_name = 'features.norm5'  # Final normalization layer after the last dense block
            elif 'resnet' in self.args.arch.lower():
                target_layer_name = 'layer4'  # For ResNet models
            elif 'vgg' in self.args.arch.lower():
                target_layer_name = 'features.42'  # For VGG19 (adjust as needed)
            else:
                target_layer_name = 'layer4'  # Default fallback
        
        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                target_layer = module
                print(f"Found target layer: {name}")
                break
        
        # If target layer not found, try architecture-specific fallbacks
        if target_layer is None:
            if 'densenet' in self.args.arch.lower():
                # For DenseNet, find the last convolutional layer in the last dense block
                for name, module in self.model.named_modules():
                    if 'denseblock4' in name and isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        print(f"Using DenseNet fallback layer: {name}")
                
                # If still not found, try the transition layer
                if target_layer is None:
                    for name, module in self.model.named_modules():
                        if 'transition3' in name and isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                            print(f"Using DenseNet transition layer: {name}")
                            
            # Add other architecture-specific fallbacks as needed
            # ...
        
        # Final fallback: use the last convolutional layer in the model
        if target_layer is None:
            print("Could not find specified target layer. Using the last convolutional layer as fallback.")
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    last_conv_name = name
            
            if target_layer:
                print(f"Using last convolutional layer: {last_conv_name}")
            else:
                raise ValueError("No convolutional layer found in the model.")
        
        # Hook for forward and backward
        activations = None
        gradients = None
        
        def save_activation(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        def save_gradient(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(save_activation)
        backward_handle = target_layer.register_full_backward_hook(save_gradient)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target class is not provided, use the predicted class
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Calculate weights based on global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Create weighted activation map
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU and normalize
        cam = torch.nn.functional.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Upsample to input size
        cam = torch.nn.functional.interpolate(
            cam, 
            size=input_tensor.shape[2:],
            mode='bilinear', 
            align_corners=False
        )
        
        return cam
    
    def process_multiple_images(self, image_paths, save_prefix="saliency"):
        """Process multiple images and visualize their saliency maps in a grid"""
        if not image_paths or len(image_paths) == 0:
            print("No image paths provided")
            return
        
        # Limit to max 3 images if we have more
        if (len(image_paths) > 3):
            print(f"Warning: Only the first 3 images will be processed (out of {len(image_paths)} provided)")
            image_paths = image_paths[:3]
            
        # Process each image and collect results
        results = []
        for img_path in image_paths:
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Warning: Image path does not exist: {img_path}")
                continue
                
            try:
                # Preprocess the image
                input_tensor, original_image = self.preprocess_image(img_path)
                
                # Get original image as numpy array
                original_np = np.array(original_image)
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1).item()
                    predicted_prob = probabilities[0, predicted_class].item()
                
                # Get class name
                class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"Class {predicted_class}"
                print(f"Image: {os.path.basename(img_path)} - Predicted class: {class_name} with probability {predicted_prob:.4f}")
                
                # Get saliency maps
                vanilla_saliency = self.compute_vanilla_saliency(input_tensor).cpu().squeeze().numpy()
                gradcam = self.compute_gradcam(input_tensor).cpu().squeeze().numpy()
                
                # Store results
                results.append({
                    'image_path': img_path,
                    'original_image': original_np,
                    'vanilla_saliency': vanilla_saliency,
                    'gradcam': gradcam,
                    'class_name': class_name,
                    'probability': predicted_prob
                })
                
                # Save individual results as before
                self.save_individual_image_results(
                    img_path, original_np, vanilla_saliency, gradcam,
                    class_name, predicted_prob, save_prefix
                )
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
        
        # Generate combined visualization if we have results
        if results and len(results) > 1:
            self.generate_combined_visualization(results, save_prefix)
    
    def save_individual_image_results(self, image_path, original, vanilla_saliency, gradcam, 
                                      class_name, probability, save_prefix):
        """Save individual saliency map results for a single image"""
        image_name = os.path.basename(image_path).split('.')[0]
        
        # Create the figure with subplots for visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image
        axes[0].imshow(original)
        axes[0].set_title(f"Original: {class_name} ({probability:.2f})")
        axes[0].axis('off')
        
        # Display vanilla saliency map
        axes[1].imshow(vanilla_saliency, cmap='hot')
        axes[1].set_title("Vanilla Gradient")
        axes[1].axis('off')
        
        # Display Grad-CAM
        axes[2].imshow(original)
        axes[2].imshow(gradcam, cmap='jet', alpha=0.5)  # Overlay with transparency
        axes[2].set_title("Grad-CAM")
        axes[2].axis('off')
        
        # Save the figure
        save_path = os.path.join(self.output_dir, f"{save_prefix}_{image_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Also save the individual components for further analysis
        # Save original image separately
        plt.figure(figsize=(5, 5))
        plt.imshow(original)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f"{save_prefix}_{image_name}_original.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save vanilla saliency separately
        plt.figure(figsize=(5, 5))
        plt.imshow(vanilla_saliency, cmap='hot')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f"{save_prefix}_{image_name}_vanilla.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save Grad-CAM separately
        plt.figure(figsize=(5, 5))
        plt.imshow(original)
        plt.imshow(gradcam, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f"{save_prefix}_{image_name}_gradcam.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_combined_visualization(self, results, save_prefix):
        """Generate a compact combined visualization for multiple images with minimal spacing"""
        # Create figure with grid of subplots
        n_images = len(results)
        
        # Use a more compact figure size with less width
        fig = plt.figure(figsize=(10, 2.5 * n_images))
        
        # Use GridSpec for more precise control of spacing
        gs = fig.add_gridspec(n_images, 3, 
                              wspace=0.0,     # Horizontal spacing between columns
                          hspace=0.03,    # Reduced from 0.03 to 0.01 for tighter vertical spacing
                          left=0.11,      
                          right=0.98,     
                          top=0.92,       # Increased from 0.88 to 0.95 to reduce top margin
                          bottom=0.02     # Reduced from 0.05 to 0.02 to reduce bottom margin
                              )
        
        axes = np.empty((n_images, 3), dtype=object)
        
        # Create each subplot with GridSpec
        for i in range(n_images):
            for j in range(3):
                axes[i, j] = fig.add_subplot(gs[i, j])
        
        # Set column titles only at the top row - smaller font and less padding
        axes[0, 0].set_title("Original", fontsize=10, pad=2)
        axes[0, 1].set_title("Vanilla Gradient", fontsize=10, pad=2)
        axes[0, 2].set_title("Grad-CAM", fontsize=10, pad=2)
        
        # Add each image and its saliency maps to the grid
        for i, result in enumerate(results):
            # Create class info text
            class_info = f"{result['class_name']} ({result['probability']:.2f})"
            
            # Original image (first column)
            axes[i, 0].imshow(result['original_image'])
            axes[i, 0].axis('off')
            
            # Add class info as vertical text on the left edge
            axes[i, 0].text(-0.15, 0.5, class_info, 
                            transform=axes[i, 0].transAxes,
                            fontsize=9, ha='right', va='center',
                            rotation=90)
            
            # Vanilla saliency (second column)
            axes[i, 1].imshow(result['vanilla_saliency'], cmap='hot')
            axes[i, 1].axis('off')
            
            # Grad-CAM (third column)
            axes[i, 2].imshow(result['original_image'])
            axes[i, 2].imshow(result['gradcam'], cmap='jet', alpha=0.5)
            axes[i, 2].axis('off')
        
        # Add dataset and model information as a smaller super title
        plt.suptitle(f"Dataset: {self.args.data}, Model: {self.args.arch}_{self.args.depth}", 
                    fontsize=12, y=0.98)
        
        # Save the figure with tight layout to minimize whitespace
        save_path = os.path.join(self.output_dir, f"{save_prefix}_combined.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Combined visualization saved to {save_path}")

    def visualize_saliency(self, image_path, save_prefix="saliency"):
        """Process a single image (maintains backward compatibility)"""
        # Just create a list with one image and use the multiple image function
        self.process_multiple_images([image_path], save_prefix)


def main():
    args = parse_args()
    generator = SaliencyMapGenerator(args)
    
    if hasattr(args, 'image_paths') and args.image_paths:
        # Process multiple images
        generator.process_multiple_images(args.image_paths)
    elif hasattr(args, 'image_path') and args.image_path:
        # Process single image (backward compatibility)
        generator.visualize_saliency(args.image_path)
    else:
        print("No images provided. Please specify images using --image_paths")
        
        # Example usage instruction
        print("\nExample usage:")
        print("python generate_saliency_maps.py --data rotc --arch densenet --depth 121 \\")
        print("  --model_path \"out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth\" \\")
        print("  --image_paths \"processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg\" \"processed_data/rotc/test/NORMAL/NORMAL-9251-2.jpeg\" \"processed_data/rotc/test/ABNORMAL/ABNORMAL-1234.jpeg\"")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import our multiprocessing fix utility
from utils.torch_utils import fix_multiprocessing_issues

# Apply the fix at the beginning before importing other modules that might use multiprocessing
fix_multiprocessing_issues()

from loader.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from gan.defense.prune import Pruner
from gan.attack.attack_loader import AttackHandler
from utils.visual.saliency_maps import SaliencyMapGenerator, compare_pruned_saliency

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_image(image_path, normalize=True, image_size=(224, 224)):
    """Load and preprocess an image"""
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
    transform = transforms.Compose(transform_list)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    return image_tensor


def load_model(args, device):
    """Load model with specified architecture and depth"""
    # Load dataset to get number of classes
    dataset_loader = DatasetLoader()
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data,
        batch_size={
            'train': 1,
            'val': 1,
            'test': 1
        }
    )
    
    # Get number of classes
    dataset = test_loader.dataset
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, 'class_to_idx'):
        num_classes = len(dataset.class_to_idx)
    else:
        raise AttributeError("Dataset does not contain class information")
        
    # Load model architecture
    model_loader = ModelLoader(device, args.arch, pretrained=False)
    models_and_names = model_loader.get_model(
        model_name=args.arch,
        depth=float(args.depth),
        input_channels=3,
        num_classes=num_classes
    )
    
    if not models_and_names:
        logging.error("No models returned from model loader")
        return None
        
    model, _ = models_and_names[0]
    model = model.to(device)
    
    # Load weights if specified
    if args.model_path:
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logging.info(f"Loaded model from {args.model_path}")
        else:
            logging.error(f"Model path {args.model_path} does not exist")
            return None
            
    model.eval()
    return model


def generate_saliency_maps(args):
    # Set device
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu")
    
    # Load model
    model = load_model(args, device)
    if model is None:
        return
        
    # Load image
    image_tensor = load_image(args.image_path)
    
    # Output directory
    output_dir = os.path.join(
        "out",
        "saliency_maps",
        args.data,
        f"{args.arch}_{args.depth}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize saliency map generator
    saliency_generator = SaliencyMapGenerator(device)
    
    # Generate different types of saliency maps for original image
    for method in args.methods:
        logging.info(f"Generating {method} saliency map...")
        if method == 'vanilla':
            saliency_map = saliency_generator.generate_vanilla_saliency(model, image_tensor)
        elif method == 'guided':
            saliency_map = saliency_generator.generate_guided_backprop(model, image_tensor)
        elif method == 'integrated':
            saliency_map = saliency_generator.generate_integrated_gradients(model, image_tensor)
        else:
            logging.warning(f"Unknown saliency method: {method}")
            continue
            
        # Save visualization
        save_path = os.path.join(output_dir, f"saliency_{method}_clean.png")
        saliency_generator.visualize_saliency(
            image_tensor, 
            saliency_map, 
            save_path=save_path,
            title=f"{args.arch}_{args.depth} - {method.title()} Saliency Map"
        )
        
    # Generate adversarial example and its saliency map if requested
    if args.generate_adversarial:
        logging.info(f"Generating adversarial example using {args.attack_type}...")
        attack_config = argparse.Namespace(
            attack_type=args.attack_type,
            attack_eps=args.attack_eps,
            attack_alpha=args.attack_eps/10,
            attack_steps=20
        )
        
        attack_handler = AttackHandler(model, args.attack_type, attack_config)
        batch_results = attack_handler.generate_adversarial_samples_batch(
            image_tensor.unsqueeze(0).to(device),
            torch.tensor([0]).to(device)  # Dummy target
        )
        
        adv_image = batch_results['adversarial'][0]
        
        for method in args.methods:
            logging.info(f"Generating {method} saliency map for adversarial example...")
            if method == 'vanilla':
                adv_saliency_map = saliency_generator.generate_vanilla_saliency(model, adv_image)
            elif method == 'guided':
                adv_saliency_map = saliency_generator.generate_guided_backprop(model, adv_image)
            elif method == 'integrated':
                adv_saliency_map = saliency_generator.generate_integrated_gradients(model, adv_image)
            else:
                continue
                
            # Save visualization
            save_path = os.path.join(output_dir, f"saliency_{method}_adversarial_{args.attack_type}.png")
            saliency_generator.visualize_saliency(
                adv_image, 
                adv_saliency_map, 
                save_path=save_path,
                title=f"{args.arch}_{args.depth} - {method.title()} Saliency Map (Adversarial: {args.attack_type})"
            )
            
    # Generate and compare saliency maps at different pruning rates
    if args.compare_pruning:
        logging.info("Comparing saliency maps at different pruning rates...")
        
        # Set pruning rates to compare
        pruning_rates = args.prune_rates or [0.0, 0.3, 0.5, 0.7, 0.9]
        
        for method in args.methods:
            compare_pruned_saliency(
                base_model=model,
                pruning_rates=pruning_rates,
                image=image_tensor,
                device=device,
                method=method,
                save_dir=output_dir,
                model_name=f"{args.arch}_{args.depth}",
                dataset_name=args.data
            )
    
    logging.info(f"Saliency maps generated and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate saliency maps for model interpretability')
    
    # Model parameters
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--arch', type=str, required=True, help='Model architecture')
    parser.add_argument('--depth', type=float, required=True, help='Model depth/variant')
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint')
    
    # Image parameters
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to image for saliency analysis')
    
    # Saliency parameters
    parser.add_argument('--methods', nargs='+', default=['vanilla', 'guided', 'integrated'],
                        help='Saliency methods to use (vanilla, guided, integrated)')
    
    # Adversarial parameters
    parser.add_argument('--generate_adversarial', action='store_true',
                        help='Generate saliency maps for adversarial examples')
    parser.add_argument('--attack_type', type=str, default='fgsm',
                        help='Attack type (fgsm, pgd, bim)')
    parser.add_argument('--attack_eps', type=float, default=0.1,
                        help='Attack epsilon/strength parameter')
    
    # Pruning parameters
    parser.add_argument('--compare_pruning', action='store_true',
                        help='Compare saliency maps at different pruning rates')
    parser.add_argument('--prune_rates', nargs='+', type=float,
                        help='Pruning rates to compare')
    
    # Other parameters
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0],
                        help='GPU IDs to use')
    
    args = parser.parse_args()
    generate_saliency_maps(args)


if __name__ == '__main__':
    main()

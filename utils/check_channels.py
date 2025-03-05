import argparse
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

def check_dataset_channels(dataset_path, num_samples=5):
    """Check the number of channels in a dataset's images"""
    # Use the same transform as our DatasetLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
    ])
    
    # Load train dataset
    train_path = Path(dataset_path) / 'train'
    if not train_path.exists():
        print(f"Path {train_path} not found")
        return
    
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Number of images: {len(train_dataset)}")
    
    # Sample a few images to check
    indices = np.random.choice(len(train_dataset), min(num_samples, len(train_dataset)), replace=False)
    
    plt.figure(figsize=(15, 3 * num_samples))
    for i, idx in enumerate(indices):
        img, label = train_dataset[idx]
        class_name = train_dataset.classes[label]
        
        # Print image stats
        print(f"Image {i+1}: Shape={img.shape}, Class={class_name}, Min={img.min()}, Max={img.max()}")
        
        # Plot image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img.permute(1, 2, 0))  # Change channel dimension for plotting
        plt.title(f"RGB Image (shape: {img.shape})")
        plt.axis('off')
        
        # Plot each channel separately
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(img[0], cmap='gray')
        plt.title(f"Channel 0 (R)")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        if img.shape[0] > 1:
            # Check if channels are identical (was a grayscale image)
            if torch.allclose(img[0], img[1]) and torch.allclose(img[1], img[2]):
                print(f"Image {i+1}: All channels are identical (converted from grayscale)")
            plt.imshow(img[1], cmap='gray')
            plt.title(f"Channel 1 (G)")
        else:
            plt.imshow(np.zeros_like(img[0]), cmap='gray')
            plt.title("No G Channel")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{Path(dataset_path).name}_channel_check.png")
    plt.close()
    
    print(f"Channel check saved to {Path(dataset_path).name}_channel_check.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset channels")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to check")
    args = parser.parse_args()
    
    check_dataset_channels(args.dataset, args.samples)


"""
To check the channels in a dataset, run the following command:

python utils/check_channels.py --dataset processed_data/chest_xray
"""
import os
import gc
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Helper Functions for Dataset Analysis
# --------------------------------------------------------------------

# Function to compute mean and std of dataset images
def compute_stats(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = 0.0
    std = 0.0
    num_images = 0

    for images, _ in tqdm(loader, desc="Computing Stats"):
        batch_samples = images.size(0)
        # Reshape to (batch, channels, height*width)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_images += batch_samples

    mean /= num_images
    std /= num_images

    return mean.numpy(), std.numpy()


# Function to analyze class distribution
def analyze_class_distribution(dataset):
    class_counts = Counter([label for _, label in dataset])
    print("\nClass Distribution Analysis:")
    print(f"Class counts: {class_counts}")
    plt.figure()
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

    # Suggestions based on class imbalance
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 5:
            print(f"Warning: Severe class imbalance detected (ratio: {imbalance_ratio:.2f}).")
            print("Suggestions: Use oversampling, undersampling, or weighted loss functions.")
        else:
            print("Class distribution is relatively balanced. No special handling required.")
    else:
        print("Only one class found. Ensure this is intentional.")


# Function to visualize sample images
def visualize_samples(dataset, num_samples=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        img, label = dataset[i]
        plt.subplot(3, 3, i + 1)
        # Convert tensor from CxHxW to HxWxC
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.show()


# Function to check image dimensions from the raw image files
def check_image_dimensions(dataset):
    original_sizes = []
    for img_path, _ in dataset.imgs:
        with Image.open(img_path) as img:
            original_sizes.append(img.size)  # (width, height)
    unique_sizes = list(set(original_sizes))
    print("\nImage Dimension Analysis:")
    if len(unique_sizes) > 10:
        print(f"Found {len(unique_sizes)} unique sizes. First 10: {unique_sizes[:10]}")
    else:
        print(f"Unique image sizes: {unique_sizes}")
    if len(unique_sizes) > 1:
        print("Warning: Images have varying dimensions. Ensure resizing or padding is applied.")
    else:
        print("All images have consistent dimensions. No resizing needed.")


# Function to check color channels from the raw image files
def check_color_channels(dataset):
    channels = []
    for img_path, _ in dataset.imgs:
        with Image.open(img_path) as img:
            channels.append(len(img.getbands()))
    unique_channels = set(channels)
    print("\nColor Channel Analysis:")
    print(f"Unique channel counts: {unique_channels}")
    if 1 in unique_channels:
        print("Warning: Grayscale images detected. Consider converting to 3-channel or using 1-channel models.")
    else:
        print("All images are RGB (3-channel). No conversion needed.")


# Function to check for corrupted files in the dataset
def check_corrupted_files(dataset):
    corrupted_files = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted_files.append(img_path)
    print("\nCorrupted File Check:")
    if corrupted_files:
        print(f"Warning: {len(corrupted_files)} corrupted files found.")
        print("Suggestions: Remove or replace corrupted files before training.")
    else:
        print("No corrupted files found.")


# Incremental pixel range analysis that avoids loading all pixel values into memory
def analyze_pixel_range(loader):
    running_max = -float("inf")
    running_min = float("inf")
    for images, _ in tqdm(loader, desc="Pixel Range Analysis"):
        batch_max = images.max().item()
        batch_min = images.min().item()
        running_max = max(running_max, batch_max)
        running_min = min(running_min, batch_min)
    print(f"Pixel range: [{running_min}, {running_max}]")
    if running_max > 1.0:
        return "0-255"
    else:
        return "0-1"


# Helper function to get unique image sizes (using raw image files)
def get_unique_image_sizes(imgs):
    sizes = set()
    for img_path, _ in imgs:
        with Image.open(img_path) as img:
            sizes.add(img.size)
    return list(sizes)


# Helper function to get the number of color channels from an image file
def get_color_channels(img_path):
    with Image.open(img_path) as img:
        return len(img.getbands())


# Helper function to count corrupted files
def count_corrupted_files(imgs):
    corrupted_count = 0
    for img_path, _ in imgs:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted_count += 1
    return corrupted_count


# Function to save results to a text file
def save_results(dataset_name, results, file_path="dataset_analysis_results.txt"):
    with open(file_path, "a") as f:
        f.write(f"\nDataset: {dataset_name}\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 50 + "\n")


# --------------------------------------------------------------------
# Main Dataset Analysis Function
# --------------------------------------------------------------------
def analyze_dataset(dataset_name, data_dir):
    try:
        print(f"\nAnalyzing dataset: {dataset_name}")

        # Define transformation for the dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Load dataset and create dataloader
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # 1. Compute image statistics
        mean, std = compute_stats(data_dir)
        print(f"Mean: {mean}, Std: {std}")

        # 2. Analyze class distribution
        analyze_class_distribution(dataset)

        # 3. Visualize sample images
        visualize_samples(dataset)
        plt.close('all')  # Clean up matplotlib

        # 4. Metadata checks
        check_image_dimensions(dataset)
        check_color_channels(dataset)
        check_corrupted_files(dataset)

        # 5. Pixel range analysis
        pixel_range = analyze_pixel_range(loader)
        plt.close('all')

        # 6. Save results using helper functions (avoiding heavy list comprehensions)
        results = {
            "Mean": mean,
            "Std": std,
            "Class Distribution": dict(Counter([label for _, label in dataset])),
            "Unique Image Sizes": get_unique_image_sizes(dataset.imgs),
            "Unique Color Channels": list({get_color_channels(img_path) for img_path, _ in dataset.imgs}),
            "Corrupted Files": count_corrupted_files(dataset.imgs),
            "Pixel Value Range": pixel_range
        }
        save_results(dataset_name, results)

        # Cleanup
        del dataset, loader, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error analyzing {dataset_name}: {str(e)}")
        with open("analysis_errors.log", "a") as f:
            f.write(f"{dataset_name} failed: {str(e)}\n")


# --------------------------------------------------------------------
# Dataset Directories and Processing
# --------------------------------------------------------------------
dataset_dirs = {
    'scisic_train': 'dataset/scisic/Train',
    'ccts_train': 'dataset/ccts/train',
    'rotc_train': 'dataset/rotc/train',
    'kvasir_train': 'dataset/kvasir/train',
    'chest_xray_train': 'dataset/chest_xray/train',
    'tbcr_train': 'dataset/tbcr',
}

# Determine the project root (modify as needed)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Process each dataset sequentially with cleanup
for dataset_name, relative_path in dataset_dirs.items():
    data_dir = os.path.join(project_root, relative_path)
    if os.path.exists(data_dir):
        analyze_dataset(dataset_name, data_dir)
    else:
        print(f"Skipping {dataset_name} - directory not found: {data_dir}")

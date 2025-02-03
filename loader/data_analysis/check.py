# check.py  ## Check Datasets before preprocessing

import os
import gc
import csv
import cv2
import torch
import numpy as np
import hashlib
from datetime import datetime
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter, defaultdict
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

# Directory to save analysis figures
FIGURES_DIR = "analysis_figures"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Define filenames for results
TXT_RESULTS_FILE = "dataset_analysis_results.txt"
CSV_RESULTS_FILE = "dataset_analysis_results.csv"


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
def analyze_class_distribution(dataset, dataset_name):
    class_counts = Counter([label for _, label in dataset])
    print("\nClass Distribution Analysis:")
    print(f"Class counts: {class_counts}")
    plt.figure()
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'{dataset_name} - Class Distribution')

    # Save figure
    class_dist_fig = os.path.join(FIGURES_DIR, f"{dataset_name}_class_distribution.png")
    plt.savefig(class_dist_fig)
    print(f"Saved class distribution figure to {class_dist_fig}")
    plt.show()

    rec = ""
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 5:
            rec = ("Warning: Severe class imbalance detected "
                   f"(ratio: {imbalance_ratio:.2f}). Consider oversampling, undersampling, or weighted loss functions.")
        else:
            rec = "Class distribution is relatively balanced. No special handling required."
    else:
        rec = "Only one class found. Ensure this is intentional."
    print(rec)
    return rec


# Function to visualize sample images and save the figure
def visualize_samples(dataset, dataset_name, num_samples=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        img, label = dataset[i]
        plt.subplot(3, 3, i + 1)
        # Convert tensor from CxHxW to HxWxC
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.suptitle(f'{dataset_name} - Sample Images', fontsize=16)
    sample_fig_path = os.path.join(FIGURES_DIR, f"{dataset_name}_sample_images.png")
    plt.savefig(sample_fig_path)
    print(f"Saved sample images figure to {sample_fig_path}")
    plt.show()


# Function to check image dimensions from the raw image files
def check_image_dimensions(dataset):
    original_sizes = []
    for img_path, _ in dataset.imgs:
        with Image.open(img_path) as img:
            original_sizes.append(img.size)  # (width, height)
    unique_sizes = list(set(original_sizes))
    msg = "\nImage Dimension Analysis:\n"
    if len(unique_sizes) > 10:
        msg += f"Found {len(unique_sizes)} unique sizes. First 10: {unique_sizes[:10]}\n"
    else:
        msg += f"Unique image sizes: {unique_sizes}\n"
    if len(unique_sizes) > 1:
        msg += "Warning: Images have varying dimensions. Ensure resizing or padding is applied.\n"
    else:
        msg += "All images have consistent dimensions. No resizing needed.\n"
    print(msg)
    return msg


# Function to check color channels from the raw image files
def check_color_channels(dataset):
    channels = []
    for img_path, _ in dataset.imgs:
        with Image.open(img_path) as img:
            channels.append(len(img.getbands()))
    unique_channels = set(channels)
    msg = "\nColor Channel Analysis:\n"
    msg += f"Unique channel counts: {unique_channels}\n"
    if 1 in unique_channels:
        msg += ("Warning: Grayscale images detected. Consider converting to 3-channel "
                "or using 1-channel models.\n")
    else:
        msg += "All images are RGB (3-channel). No conversion needed.\n"
    print(msg)
    return msg


# Function to check for corrupted files in the dataset
def check_corrupted_files(dataset):
    corrupted_files = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted_files.append(img_path)
    msg = "\nCorrupted File Check:\n"
    if corrupted_files:
        msg += f"Warning: {len(corrupted_files)} corrupted files found.\n"
        msg += "Suggestions: Remove or replace corrupted files before training.\n"
    else:
        msg += "No corrupted files found.\n"
    print(msg)
    return msg


# Incremental pixel range analysis that avoids loading all pixel values into memory
def analyze_pixel_range(loader):
    running_max = -float("inf")
    running_min = float("inf")
    for images, _ in tqdm(loader, desc="Pixel Range Analysis"):
        batch_max = images.max().item()
        batch_min = images.min().item()
        running_max = max(running_max, batch_max)
        running_min = min(running_min, batch_min)
    msg = f"Pixel range: [{running_min}, {running_max}]. "
    if running_max > 1.0:
        msg += "Pixel values are in range [0,255]. Normalization is recommended."
    else:
        msg += "Pixel values are in range [0,1]. Normalization may not be needed."
    print(msg)
    return msg


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


# 1. Duplicate Images Check (Modified to show only 2 duplicate sets)
def check_duplicate_images(dataset, limit=2, show_full=False):
    hash_dict = defaultdict(list)
    for img_path, _ in dataset.imgs:
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hash_dict[file_hash].append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Only consider keys with duplicates (i.e. count > 1)
    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    msg = ""
    if duplicates:
        msg += f"Found {len(duplicates)} sets of duplicate images. Consider removing duplicates.\n"
        count = 0
        for h, paths in duplicates.items():
            if count < limit:
                msg += f"Hash: {h} -> {len(paths)} duplicates\n"
                count += 1
            else:
                msg += "Only showing 2 duplicate sets. Use full log for more details if needed.\n"
                break
        # Optional: To show full details, set show_full=True (uncomment below if desired)
        # if show_full:
        #     msg = f"Found {len(duplicates)} sets of duplicate images. Details:\n"
        #     for h, paths in duplicates.items():
        #         msg += f"Hash: {h} -> {len(paths)} duplicates: {paths}\n"
    else:
        msg = "No duplicate images found.\n"
    print(msg)
    return msg


# 2. Noise and Artifact Detection (using Laplacian variance)
def detect_noise_and_artifacts(dataset):
    laplacian_vars = []
    for img_path, _ in dataset.imgs:
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                laplacian_vars.append(lap_var)
        except Exception as e:
            print(f"Error processing {img_path} for noise: {e}")

    if laplacian_vars:
        avg_lap = np.mean(laplacian_vars)
        msg = f"Average Laplacian variance (noise/sharpness measure): {avg_lap:.2f}\n"
        if avg_lap < 50:
            msg += "Warning: Images may be too blurry/noisy. Consider enhancing image quality.\n"
        else:
            msg += "Image sharpness/noise levels appear acceptable.\n"
    else:
        msg = "Could not compute noise/artifact metrics.\n"
    print(msg)
    return msg


# 3. Contrast and Brightness Analysis
def analyze_contrast_and_brightness(dataset):
    brightness_vals = []
    contrast_vals = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness_vals.append(np.mean(arr))
                contrast_vals.append(np.std(arr))
        except Exception as e:
            print(f"Error processing {img_path} for brightness/contrast: {e}")

    if brightness_vals and contrast_vals:
        avg_brightness = np.mean(brightness_vals)
        avg_contrast = np.mean(contrast_vals)
        msg = (f"Average brightness: {avg_brightness:.2f}\n"
               f"Average contrast: {avg_contrast:.2f}\n")
        if avg_brightness < 50:
            msg += "Warning: Overall brightness is low. Consider image enhancement.\n"
        elif avg_brightness > 200:
            msg += "Warning: Overall brightness is very high. Check for overexposure.\n"
        else:
            msg += "Brightness appears within acceptable range.\n"
        if avg_contrast < 30:
            msg += "Warning: Low contrast detected. Consider contrast enhancement.\n"
        else:
            msg += "Contrast levels appear acceptable.\n"
    else:
        msg = "Could not analyze brightness/contrast.\n"
    print(msg)
    return msg


# 4. File Format and Metadata Consistency
def check_file_format_and_metadata(dataset):
    formats = []
    exif_errors = []
    for img_path, _ in dataset.imgs:
        ext = os.path.splitext(img_path)[1].lower()
        formats.append(ext)
        try:
            with Image.open(img_path) as img:
                # Attempt to get EXIF data if available
                exif = img._getexif()
        except Exception as e:
            exif_errors.append(img_path)
    format_counts = Counter(formats)
    msg = "File format distribution:\n"
    for fmt, count in format_counts.items():
        msg += f"  {fmt}: {count}\n"
    if exif_errors:
        msg += f"Warning: Could not extract metadata from {len(exif_errors)} images.\n"
    else:
        msg += "Metadata extraction appears consistent.\n"
    print(msg)
    return msg


# 5. Resolution and Aspect Ratio
def analyze_resolution_and_aspect_ratio(dataset):
    aspect_ratios = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if h != 0:
                    aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Error processing {img_path} for aspect ratio: {e}")
    if aspect_ratios:
        mean_ratio = np.mean(aspect_ratios)
        min_ratio = np.min(aspect_ratios)
        max_ratio = np.max(aspect_ratios)
        msg = (f"Aspect Ratio - Mean: {mean_ratio:.2f}, "
               f"Min: {min_ratio:.2f}, Max: {max_ratio:.2f}\n")
        if max_ratio - min_ratio > 0.5:
            msg += ("Warning: Significant variation in aspect ratios. "
                    "Consider standardizing cropping or padding.\n")
        else:
            msg += "Aspect ratios appear consistent.\n"
    else:
        msg = "Could not compute aspect ratios.\n"
    print(msg)
    return msg


# 6. Data Augmentation Feasibility
def evaluate_data_augmentation_feasibility(dataset):
    brightness_vals = []
    contrast_vals = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness_vals.append(np.mean(arr))
                contrast_vals.append(np.std(arr))
        except Exception as e:
            print(f"Error processing {img_path} for augmentation feasibility: {e}")
    if brightness_vals and contrast_vals:
        brightness_range = np.max(brightness_vals) - np.min(brightness_vals)
        contrast_range = np.max(contrast_vals) - np.min(contrast_vals)
        msg = (f"Brightness range: {brightness_range:.2f}\n"
               f"Contrast range: {contrast_range:.2f}\n")
        if brightness_range < 30 or contrast_range < 10:
            msg += ("Warning: Limited variation in brightness/contrast. "
                    "Data augmentation (e.g., brightness/contrast adjustments) is recommended.\n")
        else:
            msg += "Diversity in brightness/contrast appears sufficient.\n"
    else:
        msg = "Could not evaluate data augmentation feasibility.\n"
    print(msg)
    return msg


# Function to save results to a text file (with timestamp)
def save_results(dataset_name, results, file_path=TXT_RESULTS_FILE):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as f:
        f.write(f"\nTimestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 50 + "\n")
    print(f"Results saved to {file_path}")


# Function to save results to a CSV file (with timestamp)
def save_results_csv(dataset_name, results, file_path=CSV_RESULTS_FILE):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = ["Dataset", "Timestamp", "Mean", "Std", "Class Distribution",
              "Unique Image Sizes", "Unique Color Channels", "Corrupted Files", "Pixel Value Range",
              "Duplicate Check", "Noise/Artifacts", "Brightness/Contrast", "File Format/Metadata",
              "Aspect Ratio", "Data Augmentation Feasibility"]

    row = {
        "Dataset": dataset_name,
        "Timestamp": timestamp,
        "Mean": results.get("Mean"),
        "Std": results.get("Std"),
        "Class Distribution": results.get("Class Distribution"),
        "Unique Image Sizes": results.get("Unique Image Sizes"),
        "Unique Color Channels": results.get("Unique Color Channels"),
        "Corrupted Files": results.get("Corrupted Files"),
        "Pixel Value Range": results.get("Pixel Value Range"),
        "Duplicate Check": results.get("Duplicate Check"),
        "Noise/Artifacts": results.get("Noise/Artifacts"),
        "Brightness/Contrast": results.get("Brightness/Contrast"),
        "File Format/Metadata": results.get("File Format/Metadata"),
        "Aspect Ratio": results.get("Aspect Ratio"),
        "Data Augmentation Feasibility": results.get("Data Augmentation Feasibility")
    }

    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to CSV file {file_path}")


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

        # 2. Analyze class distribution (and save the plot)
        class_dist_rec = analyze_class_distribution(dataset, dataset_name)

        # 3. Visualize sample images (and save the plot)
        visualize_samples(dataset, dataset_name)
        plt.close('all')  # Clean up matplotlib

        # 4. Metadata checks
        dims_msg = check_image_dimensions(dataset)
        color_msg = check_color_channels(dataset)
        corrupted_msg = check_corrupted_files(dataset)

        # 5. Pixel range analysis
        pixel_range_msg = analyze_pixel_range(loader)
        plt.close('all')

        # 6. Additional Checks
        dup_msg = check_duplicate_images(dataset)
        noise_msg = detect_noise_and_artifacts(dataset)
        brightness_contrast_msg = analyze_contrast_and_brightness(dataset)
        file_meta_msg = check_file_format_and_metadata(dataset)
        aspect_ratio_msg = analyze_resolution_and_aspect_ratio(dataset)
        aug_msg = evaluate_data_augmentation_feasibility(dataset)

        # 7. Compile all results
        results = {
            "Mean": mean,
            "Std": std,
            "Class Distribution": class_dist_rec,
            "Image Dimensions": dims_msg,
            "Color Channels": color_msg,
            "Corrupted Files": corrupted_msg,
            "Pixel Value Range": pixel_range_msg,
            "Duplicate Check": dup_msg,
            "Noise/Artifacts": noise_msg,
            "Brightness/Contrast": brightness_contrast_msg,
            "File Format/Metadata": file_meta_msg,
            "Aspect Ratio": aspect_ratio_msg,
            "Data Augmentation Feasibility": aug_msg
        }

        save_results(dataset_name, results)
        save_results_csv(dataset_name, results)

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

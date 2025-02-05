# preprocessing.py
import os
import logging
import torch
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional, Union, cast
from collections.abc import Sized
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import cv2
import yaml


# -----------------------------------------------------------------------------
# Config Loader
# -----------------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    """
    Loads a YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_preprocessing_config(config_path: str = "./loader/config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("datasets", {})


# -----------------------------------------------------------------------------
# Additional Preprocessing Functions
# -----------------------------------------------------------------------------
def apply_hu_window(image: Image.Image, window: Tuple[int, int] = (-1000, 400)) -> Image.Image:
    """
    For CT scans: apply Hounsfield unit (HU) windowing.
    """
    image_np = np.array(image).astype(np.float32)
    lower, upper = window
    image_np = np.clip(image_np, lower, upper)
    # Normalize to [0, 1]
    image_np = (image_np - lower) / (upper - lower)
    # Convert back to 8-bit image
    return Image.fromarray((image_np * 255).astype(np.uint8))


def apply_clahe(image: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
    """
    image_np = np.array(image)
    # If image is grayscale
    if len(image_np.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image_clahe = clahe.apply(image_np)
        return Image.fromarray(image_clahe)
    else:
        # For color images, convert to LAB and apply CLAHE on the L channel.
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        image_lab = cv2.merge((l, a, b))
        image_clahe = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image_clahe)


def apply_median_filter(image: Image.Image, kernel_size=3) -> Image.Image:
    """
    Apply a median filter to remove salt-and-pepper noise.
    """
    image_np = np.array(image)
    filtered = cv2.medianBlur(image_np, kernel_size)
    return Image.fromarray(filtered)


def apply_gaussian_filter(image: Image.Image, sigma=1.0) -> Image.Image:
    """
    Apply a Gaussian filter to reduce noise.
    """
    image_np = np.array(image)
    filtered = cv2.GaussianBlur(image_np, (0, 0), sigma)
    return Image.fromarray(filtered)


def apply_histogram_equalization(image: Image.Image) -> Image.Image:
    """
    Apply histogram equalization to a grayscale image.
    """
    gray = image.convert("L")
    image_np = np.array(gray)
    eq = cv2.equalizeHist(image_np)
    return Image.fromarray(eq)


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def get_default_transforms(preproc_config: dict = None):
    """
    Returns a dictionary of transforms for 'train', 'val', and 'test' sets.
    If preproc_config is provided (a dict from YAML), its parameters are used.
    """
    # Determine normalization method from config (default: minmax to [-1, 1])
    if preproc_config is not None and "normalize" in preproc_config:
        norm_conf = preproc_config["normalize"]
        # You can add more options here
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # Build a list of additional preprocessing steps based on the config.
    additional_steps = []
    if preproc_config is not None:
        if "windowing" in preproc_config:
            win_conf = preproc_config["windowing"]
            additional_steps.append(
                transforms.Lambda(lambda img: apply_hu_window(img, window=tuple(win_conf.get("window", [-1000, 400]))))
            )
        if "artifact_removal" in preproc_config:
            art_conf = preproc_config["artifact_removal"]
            if art_conf["method"] == "median_filter":
                additional_steps.append(
                    transforms.Lambda(lambda img: apply_median_filter(img, kernel_size=art_conf.get("kernel_size", 3)))
                )
            elif art_conf["method"] == "gaussian_filter":
                additional_steps.append(
                    transforms.Lambda(lambda img: apply_gaussian_filter(img, sigma=art_conf.get("sigma", 1.0)))
                )
        if "contrast_enhancement" in preproc_config:
            ce_conf = preproc_config["contrast_enhancement"]
            if ce_conf["method"] == "CLAHE":
                additional_steps.append(
                    transforms.Lambda(lambda img: apply_clahe(img,
                                                              clip_limit=ce_conf.get("clip_limit", 2.0),
                                                              tile_grid_size=tuple(
                                                                  ce_conf.get("tile_grid_size", [8, 8]))))
                )
            elif ce_conf["method"] == "histogram_equalization":
                additional_steps.append(
                    transforms.Lambda(lambda img: apply_histogram_equalization(img))
                )

    # Define the full transform pipelines:
    data_transforms = {
        'train': transforms.Compose(
            additional_steps +
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                normalize
            ]
        ),
        'val': transforms.Compose(
            additional_steps +
            [
                transforms.CenterCrop(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                normalize
            ]
        ),
        'test': transforms.Compose(
            additional_steps +
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]
        )
    }
    return data_transforms


def build_transforms(dataset_name: str, mode: str = "train"):
    """
    Build a transformation pipeline based on the YAML config for the given dataset.
    mode: "train", "val", or "test"
    """
    config = load_preprocessing_config()
    ds_cfg = config.get(dataset_name, {})
    transform_list = []

    # Conversion: if dataset includes grayscale images that need to be converted
    if ds_cfg.get("conversion", "") == "convert_to_3_channel":
        transform_list.append(transforms.Lambda(lambda img: img.convert("RGB") 
                              if img.mode != "RGB" else img))
    
    # Resize and/or Padding
    if ds_cfg.get("resize", False):
        target_size = tuple(ds_cfg.get("target_size", [224, 224]))
        if ds_cfg.get("padding", False):
            transform_list.append(transforms.Resize(target_size))
            transform_list.append(transforms.Pad(10))  # Example fixed padding; adjust as needed
        else:
            transform_list.append(transforms.Resize(target_size))

    # Data augmentation for training
    if mode == "train" and ds_cfg.get("augmentation", {}).get("contrast_enhancement", False):
        transform_list.append(transforms.ColorJitter(contrast=0.5))

    # Convert image to tensor
    transform_list.append(transforms.ToTensor())

    # Normalization
    if "normalization" in ds_cfg:
        norm_cfg = ds_cfg["normalization"]
        transform_list.append(transforms.Normalize(mean=norm_cfg.get("mean", [0.5, 0.5, 0.5]),
                                                    std=norm_cfg.get("std", [0.5, 0.5, 0.5])))

    return transforms.Compose(transform_list)

# Example function to preprocess a dataset directory
def preprocess_dataset(dataset_dir: str, dataset_name: str, mode: str = "train"):
    transform = build_transforms(dataset_name, mode)
    # For example, use ImageFolder with the transform pipeline.
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root=dataset_dir, transform=transform)
    return dataset


# -----------------------------------------------------------------------------
# Outlier Removal Methods
# -----------------------------------------------------------------------------
def remove_outliers_isolation_forest(X, contamination=0.1):
    n_samples, channels, height, width = X.shape
    X_reshaped = X.reshape(n_samples, -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(X_scaled)
    outliers = iso_forest.predict(X_scaled)
    outlier_indices = np.where(outliers == -1)[0]
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


def remove_outliers_lof(X, n_neighbors=20, contamination=0.1):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof.fit_predict(X_scaled)
    outlier_indices = np.where(outliers == -1)[0]
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


def remove_outliers_dbscan(X, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    outlier_indices = np.where(clusters == -1)[0]
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


# -----------------------------------------------------------------------------
# Dataset Splitting and Sampler
# -----------------------------------------------------------------------------
def split_dataset(dataset: Union[Dataset, Tuple[Dataset, ...], Sized]) -> Tuple[
    Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    if dataset is None or len(dataset) == 0:
        return None, None, None
    if isinstance(dataset, tuple):
        if all(isinstance(d, (Dataset, type(None))) for d in dataset):
            return cast(Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]], dataset)
        else:
            raise ValueError("All elements must be torch.utils.data.Dataset instances or None.")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    if train_size > 0 and val_size > 0 and test_size > 0:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
    else:
        raise ValueError("Dataset is too small to be split into train, validation, and test sets.")


def get_WeightedRandom_Sampler(subset_dataset, original_dataset):
    original_dataset = original_dataset.dataset if isinstance(original_dataset,
                                                              torch.utils.data.Subset) else original_dataset
    dataLoader = DataLoader(subset_dataset, batch_size=512)
    All_target = []
    for _, (_, targets) in enumerate(dataLoader):
        for i in range(targets.shape[0]):
            All_target.append(targets[i].item())
    target = np.array(All_target)
    logging.info("\nClass distribution in the dataset:")
    for i, class_name in enumerate(original_dataset.classes):
        logging.info(f"{np.sum(target == i)}: {class_name}")
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight).double()
    Sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return Sampler


def get_dataloader_target_class_number(dataLoader):
    original_dataset = dataLoader.dataset
    if isinstance(original_dataset, torch.utils.data.Subset):
        original_dataset = original_dataset.dataset
    All_target_2 = []
    for batch_idx, (inputs, targets) in enumerate(dataLoader):
        for i in range(targets.shape[0]):
            All_target_2.append(targets[i].item())
    data = np.array(All_target_2)
    unique_classes, counts = np.unique(data, return_counts=True)
    logging.info("Unique classes and their counts in the dataset:")
    for cls, count in zip(unique_classes, counts):
        logging.info(f"{count}: {original_dataset.classes[cls]}")
    return original_dataset.classes, len(original_dataset.classes)


def check_for_corrupted_images(directory, transform):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = transform(img)
                except Exception as e:
                    logging.error(f"Corrupted image file: {img_path} - {e}")


# -----------------------------------------------------------------------------
# Dataset Preprocessing
# -----------------------------------------------------------------------------
def preprocess_dataset(train_dataset):
    # Convert dataset to a NumPy array of images
    train_data = np.array([np.array(img) for img, _ in train_dataset])
    train_labels = np.array([label for _, label in train_dataset])
    # Remove outliers using Isolation Forest (or you could choose another method)
    train_data_cleaned = remove_outliers_isolation_forest(train_data)
    # Convert cleaned images back to PIL Images and recreate dataset
    train_dataset_cleaned = [
        (transforms.ToPILImage()(img.permute(1, 2, 0).numpy().astype(np.uint8)), label)
        for img, label in train_data_cleaned
    ]
    return train_dataset_cleaned

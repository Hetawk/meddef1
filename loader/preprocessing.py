# preprocessing.py
import os
import logging
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Optional, Union, cast
from collections.abc import Sized
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import cv2
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


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
    config = load_config(config_path)
    return config.get("data", {})


# Add this function after the config loading functions
def get_default_transforms(preproc_config: dict = None):
    """
    Returns a dictionary of transforms for 'train', 'val', and 'test' sets.
    If preproc_config is provided (a dict from YAML), its parameters are used.
    """
    # Use config to determine resize value; default to [224, 224]
    resize = preproc_config.get(
        "resize", [224, 224]) if preproc_config else [224, 224]
    train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    return {"train": train, "val": val, "test": test}


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
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        image_clahe = clahe.apply(image_np)
        return Image.fromarray(image_clahe)
    else:
        # For color images, convert to LAB and apply CLAHE on the L channel.
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
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


def apply_padding(image: Image.Image, padding_config: dict) -> Image.Image:
    """
    Apply padding based on configuration.
    """
    if not padding_config.get('enabled', False):
        return image

    strategy = padding_config.get('strategy', 'symmetric')
    if strategy == 'symmetric':
        return ImageOps.expand(image, border=padding_config.get('value', 10), fill=0)
    elif strategy == 'constant':
        return transforms.Pad(padding_config.get('value', 0))(image)
    return image


def apply_brightness_adjustment(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Adjust image brightness.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_padding_with_strategy(image: Image.Image, padding_config: dict) -> Image.Image:
    """
    Enhanced padding function that supports multiple strategies from config:
    - symmetric: Equal padding on all sides
    - constant: Padding with constant value
    - reflect: Mirror padding
    """
    if not padding_config.get('enabled', False):
        return image

    strategy = padding_config.get('strategy', 'symmetric')
    pad_value = padding_config.get('value', 0)

    if strategy == 'symmetric':
        # Calculate padding to make image square if pad_to_square is True
        if padding_config.get('pad_to_square', False):
            w, h = image.size
            max_dim = max(w, h)
            delta_w = max_dim - w
            delta_h = max_dim - h
            padding = (delta_w//2, delta_h//2, delta_w -
                       (delta_w//2), delta_h-(delta_h//2))
            return ImageOps.expand(image, padding, fill=pad_value)
        return ImageOps.expand(image, border=pad_value, fill=0)
    elif strategy == 'reflect':
        return ImageOps.expand(image, border=pad_value, fill=ImageOps.REFLECT)
    else:  # constant
        return ImageOps.expand(image, border=pad_value, fill=pad_value)


def apply_image_enhancement(image: Image.Image, enhancement_config: dict) -> Image.Image:
    """
    Apply multiple image enhancements based on config
    """
    if enhancement_config.get('brightness_adjustment', False):
        factor = enhancement_config.get('brightness_factor', 1.2)
        image = ImageEnhance.Brightness(image).enhance(factor)

    if enhancement_config.get('contrast_enhancement', False):
        factor = enhancement_config.get('contrast_factor', 1.5)
        image = ImageEnhance.Contrast(image).enhance(factor)

    return image


def apply_noise_reduction(image: Image.Image, config: dict) -> Image.Image:
    """
    Apply noise reduction based on config settings
    """
    method = config.get('noise_reduction_method', 'gaussian')
    if method == 'gaussian':
        return image.filter(ImageFilter.GaussianBlur(radius=1))
    elif method == 'median':
        return image.filter(ImageFilter.MedianFilter(size=3))
    return image


class DatasetPreprocessor:
    """New class to handle dataset-specific preprocessing"""

    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config
        self.dataset_config = self._get_dataset_config()
        self.structure_type = self._get_structure_type()

    def _get_dataset_config(self):
        return next((d for d in self.config['data_key']
                    if d['name'] == self.dataset_name), None)

    def _get_structure_type(self):
        return self.dataset_config['structure'].get('type', 'standard')

    def preprocess(self, data_path: Path, mode: str) -> Dataset:
        """Main preprocessing pipeline"""
        transforms_list = self._build_transform_pipeline(mode)
        if self.structure_type == "class_based":
            return self._preprocess_class_based(data_path, transforms_list)
        return preprocess_dataset(str(data_path), self.dataset_name, mode)

    def _build_transform_pipeline(self, mode: str):
        """Build transforms with fallback to default transforms"""
        config = load_preprocessing_config()
        ds_cfg = config.get(self.dataset_name, {})
        common_cfg = config.get('common_settings', {})

        # Get default transforms as fallback
        default_transforms = get_default_transforms(common_cfg)

        # If no special processing needed, return default transforms
        if not ds_cfg.get('preprocessing') and not ds_cfg.get('augmentation'):
            return default_transforms[mode]

        # Otherwise build custom transform pipeline
        transform_list = []

        # 1. Padding (with enhanced strategies)
        padding_config = {
            **common_cfg.get('padding', {}), **ds_cfg.get('padding', {})}
        if padding_config.get('enabled', False):
            transform_list.append(
                transforms.Lambda(
                    lambda img: apply_padding_with_strategy(img, padding_config))
            )

        # 2. Channel conversion
        if ds_cfg.get("conversion", "") == "convert_to_3_channel":
            transform_list.append(
                transforms.Lambda(lambda img: img.convert("RGB"))
            )

        # 3. Base resize
        resize_size = ds_cfg.get(
            "resize", common_cfg.get("resize", [224, 224]))
        transform_list.append(transforms.Resize(resize_size))

        # 4. Training mode specific transforms
        if mode == "train":
            # Apply preprocessing before augmentation
            preproc_cfg = ds_cfg.get('preprocessing', {})
            if preproc_cfg.get('clahe', False):
                transform_list.append(
                    transforms.Lambda(lambda img: apply_clahe(img))
                )
            if preproc_cfg.get('remove_noise', False):
                transform_list.append(
                    transforms.Lambda(
                        lambda img: apply_noise_reduction(img, preproc_cfg))
                )

            # Augmentation pipeline
            aug_cfg = ds_cfg.get('augmentation', {})
            if aug_cfg:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: apply_image_enhancement(img, aug_cfg))
                )
                # Add random transforms for training
                if aug_cfg.get('random_horizontal_flip', True):
                    transform_list.append(transforms.RandomHorizontalFlip())
                if aug_cfg.get('random_rotation', False):
                    transform_list.append(transforms.RandomRotation(10))

        # 5. To Tensor and Normalization
        transform_list.append(transforms.ToTensor())
        if "normalization" in ds_cfg:
            norm_cfg = ds_cfg["normalization"]
            transform_list.append(transforms.Normalize(
                mean=norm_cfg['mean'],
                std=norm_cfg['std']
            ))

        return transforms.Compose(transform_list)


def build_transforms(dataset_name: str, mode: str = "train"):
    """Updated to handle all dataset structures"""
    config = load_preprocessing_config()
    preprocessor = DatasetPreprocessor(dataset_name, config)
    return preprocessor._build_transform_pipeline(mode)


def preprocess_dataset(dataset_dir: str, dataset_name: str, mode: str = "train") -> Dataset:
    """Updated to handle different dataset structures"""
    config = load_preprocessing_config()
    preprocessor = DatasetPreprocessor(dataset_name, config)
    return preprocessor.preprocess(Path(dataset_dir), mode)


# Add new helper functions for specific preprocessing tasks
def process_metadata(metadata_file: Path) -> dict:
    """Process dataset metadata files"""
    if metadata_file.suffix == '.xlsx':
        return pd.read_excel(metadata_file)
    return pd.read_csv(metadata_file)


def validate_dataset_structure(dataset_path: Path, expected_structure: dict) -> bool:
    """Validate dataset folder structure"""
    if not dataset_path.exists():
        return False

    structure_type = expected_structure.get('type', 'standard')

    if structure_type == 'class_based':
        return all(
            (dataset_path / class_name).exists()
            for class_name in expected_structure['classes']
        )

    required_dirs = {
        'standard': ['train', 'val', 'test'],
        'train_test': ['train', 'test'],
        'train_valid_test': ['train', 'valid', 'test']
    }

    return all(
        (dataset_path / dir_name).exists()
        for dir_name in required_dirs.get(structure_type, [])
    )


# Update split_dataset function to handle different structures
def split_dataset(
    dataset: Union[Dataset, Tuple[Dataset, ...], Sized],
    split_ratios: List[float] = None,
    structure_type: str = 'standard'
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Enhanced split function that handles different dataset structures"""
    if dataset is None or len(dataset) == 0:
        return None, None, None

    if isinstance(dataset, tuple):
        if all(isinstance(d, (Dataset, type(None))) for d in dataset):
            return cast(Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]], dataset)
        raise ValueError(
            "All elements must be torch.utils.data.Dataset instances or None.")

    if not split_ratios:
        split_ratios = [0.7, 0.15, 0.15]  # default ratios

    if structure_type == 'train_test':
        # Split train into train/val
        train_size = split_ratios[0] / (split_ratios[0] + split_ratios[1])
        train_data, val_data = train_test_split(
            dataset, train_size=train_size, random_state=42)
        return train_data, val_data, None

    # Standard three-way split
    train_size = int(split_ratios[0] * len(dataset))
    val_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if train_size > 0 and val_size > 0 and test_size > 0:
        return random_split(dataset, [train_size, val_size, test_size])

    raise ValueError("Dataset too small for splitting")


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
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination)
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
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
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

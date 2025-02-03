# preprocessing.py
import os

import torch
from PIL.Image import Image
from torchvision import transforms
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional, Union, cast
from collections.abc import Sized
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import logging




def get_default_transforms():
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std (not ideal for medical images)
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize]),
        'val': transforms.Compose([
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]),
    }
    return data_transforms


# Outlier 1: Isolation Forest
def remove_outliers_isolation_forest(X, contamination=0.1):
    # Reshape the data to 2D
    n_samples, channels, height, width = X.shape
    X_reshaped = X.reshape(n_samples, -1)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(X_scaled)

    # Predict outliers
    outliers = iso_forest.predict(X_scaled)
    outlier_indices = np.where(outliers == -1)[0]

    # Remove outliers from the dataset
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


# Outlier 2: Local Outlier Factor
def remove_outliers_lof(X, n_neighbors=20, contamination=0.1):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the LOF model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof.fit_predict(X_scaled)
    outlier_indices = np.where(outliers == -1)[0]

    # Remove outliers from the dataset
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


# Outlier 3: Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
def remove_outliers_dbscan(X, eps=0.5, min_samples=5):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    outlier_indices = np.where(clusters == -1)[0]

    # Remove outliers from the dataset
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


def split_dataset(dataset: Union[Dataset, Tuple[Dataset, ...], Sized]) -> Tuple[
    Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """
    Splits the dataset into training, validation, and test sets based on specified proportions.
    If the dataset cannot be split (e.g., it is None or empty), returns a tuple of (None, None, None).

    """
    if dataset is None or len(dataset) == 0:
        return None, None, None

    if isinstance(dataset, tuple):
        # Ensure all elements are instances of Dataset or None
        if all(isinstance(d, (Dataset, type(None))) for d in dataset):
            # Explicitly cast to the expected tuple type
            return cast(Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]], dataset)
        else:
            raise ValueError("All elements of the input tuple must be torch.utils.data.Dataset instances or None.")

    train_size = int(0.7 * len(dataset))  # 70% of the dataset for training
    val_size = int(0.15 * len(dataset))  # 15% of the dataset for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for testing

    # Ensure there is enough data to split according to the specified proportions
    if train_size > 0 and val_size > 0 and test_size > 0:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
    else:
        raise ValueError("Dataset is too small to be split into train, "
                         "validation, and test sets with the specified proportions.")

def get_input_channels(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
    """
    Retrieves the number of input channels from the training dataset.

    Args:
        train_batch_size (int): Batch size for the training DataLoader.
        val_batch_size (int): Batch size for the validation DataLoader.
        test_batch_size (int): Batch size for the test DataLoader.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to use pinned memory for data loading.

    Returns:
        int: The number of input channels in the dataset.
    """
    train_loader, val_loader, test_loader = self.load(train_batch_size, val_batch_size, test_batch_size,
                                                      num_workers, pin_memory)
    train_dataset = train_loader.dataset
    sample_img, _ = train_dataset[0]
    return sample_img.shape[0]

def get_WeightedRandom_Sampler(subset_dataset, original_dataset):
    # Access the original dataset from the Subset object
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
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    Sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return Sampler

def get_dataloader_target_class_number(dataLoader):
    # Access the original dataset from the Subset object
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
    """
    Fix truncated images
    Checks for corrupted images in the specified directory.

    Args:
        directory (str): The directory to check for corrupted images.
        transform (callable): The transform to apply to the images.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = transform(img)
                except Exception as e:
                    logging.error(f"Corrupted image file: {img_path} - {e}")

# Preprocess dataset
def preprocess_dataset(train_dataset):
    # Extract data to a NumPy array
    train_data = np.array([np.array(img) for img, _ in train_dataset])
    train_labels = np.array([label for _, label in train_dataset])

    # Apply outlier removal
    train_data_cleaned = remove_outliers_isolation_forest(train_data)

    # Convert cleaned data back to a dataset
    train_dataset_cleaned = [
        (transforms.ToPILImage()(img.permute(1, 2, 0).numpy().astype(np.uint8)), label)
        for img, label in train_data_cleaned
    ]

    return train_dataset_cleaned

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import logging
from typing import Dict, Tuple
import os
from PIL import Image


class DatasetLoader:
    _instance = None
    SUPPORTED_DATASETS = {
        'ccts', 'scisic', 'rotc', 'kvasir', 'dermnet',
        'chest_xray', 'tbcr', 'miccai_brats2020'
    }

    def __init__(self):
        self.processed_data_path = Path("processed_data")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def validate_dataset(self, dataset_name: str) -> bool:
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Available datasets: {self.SUPPORTED_DATASETS}")

        dataset_path = self.processed_data_path / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {dataset_path}")

        # Validate that all required directories exist
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if not (dataset_path / split).exists():
                raise FileNotFoundError(
                    f"Split directory '{split}' missing in {dataset_path}")

        # Check that each split has some data
        for split in required_splits:
            split_path = dataset_path / split
            if not any(split_path.iterdir()):
                logging.warning(
                    f"Split directory '{split}' is empty in {dataset_path}")

        # Check for channel consistency across splits
        try:
            splits = ['train', 'val', 'test']
            channels = []
            for split in splits:
                split_path = dataset_path / split
                if split_path.exists():
                    # Find first image to check channels
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir():
                            for img_path in class_dir.glob('*.*'):
                                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                                    with Image.open(img_path) as img:
                                        channels.append(
                                            (split, len(img.getbands()), img.getbands()))
                                    break
                            if channels and channels[-1][0] == split:
                                break

            # Check if all splits have the same number of channels
            if len(set(ch[1] for ch in channels)) > 1:
                channel_info = ", ".join(
                    f"{s}: {c} {b}" for s, c, b in channels)
                logging.warning(
                    f"Inconsistent channel counts detected in {dataset_name}: {channel_info}")
        except Exception as e:
            logging.warning(f"Could not verify channel consistency: {e}")

        return True

    def load_data(self, dataset_name: str, batch_size: Dict[str, int],
                  num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load processed datasets"""
        self.validate_dataset(dataset_name)
        dataset_path = self.processed_data_path / dataset_name

        train_dataset = ImageFolder(dataset_path / 'train', self.transform)
        val_dataset = ImageFolder(dataset_path / 'val', self.transform)
        test_dataset = ImageFolder(dataset_path / 'test', self.transform)

        logging.info(f"Loading {dataset_name} dataset:")
        logging.info(f"Found {len(train_dataset)} training samples")
        logging.info(f"Found {len(val_dataset)} validation samples")
        logging.info(f"Found {len(test_dataset)} test samples")
        logging.info(f"Classes: {train_dataset.classes}")

        # Check for potentially problematic split sizes
        if len(val_dataset) < 10:
            logging.warning(f"Validation set is very small ({len(val_dataset)} samples). "
                            "Consider using --enforce_split option with dataset_processing.py")
        if len(test_dataset) < 10:
            logging.warning(
                f"Test set is very small ({len(test_dataset)} samples).")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size['train'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size['val'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

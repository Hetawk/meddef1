from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import logging
from typing import Dict, Tuple


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

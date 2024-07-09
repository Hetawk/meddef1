# dataset_loader.py

import os
import logging

import numpy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import nibabel as nib
import types


class DatasetLoader:

    def __init__(self, dataset_name, data_dir='./dataset'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.datasets_dict = {
            # 'mnist': self.load_mnist,
            # 'med': self.load_med,
            'ccts': self.load_ccts,
            # 'tbcr': self.load_tbcr,
            # 'scisic': self.load_scisic,

            # Add more datasets here...
        }
        if self.dataset_name not in self.datasets_dict:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")
        logging.info(f"DatasetLoader initialized for {dataset_name}.")

    @staticmethod
    def get_all_datasets(data_dir='./dataset'):
        return {
            # 'mnist': DatasetLoader('mnist', data_dir),
            # 'med': DatasetLoader('med', data_dir),
            'ccts': DatasetLoader('ccts', data_dir),
            # 'tbcr': DatasetLoader('tbcr', data_dir),
            # 'scisic': DatasetLoader('scisic', data_dir),
        }

    def load(self):
        logging.info(f"Loading dataset: {self.dataset_name}.")
        try:
            train_dataset, val_dataset, test_dataset, _ = self.datasets_dict[self.dataset_name]()
            return train_dataset, val_dataset, test_dataset
        except KeyError:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")

    # Dataset #1: MNIST
    def load_mnist(self):
        train_dataset = datasets.MNIST(os.path.join(self.data_dir, 'mnist'), train=True, download=True)
        test_dataset = datasets.MNIST(os.path.join(self.data_dir, 'mnist'), train=False, download=True)
        train_dataset, val_dataset, _ = self.split_dataset(train_dataset)  # Adjusted unpacking
        return train_dataset, test_dataset, val_dataset, train_dataset.classes

    # Dataset #2: Med
    def load_med(self):
        train_dir = os.path.join(self.data_dir, 'med')
        full_dataset = self.CustomImageDataset(train_dir)
        train_dataset, val_dataset, test_dataset = self.split_dataset(full_dataset)
        return train_dataset, test_dataset, val_dataset, train_dataset.classes

    # Dataset #3: scisic
    def load_scisic(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(self.data_dir, 'scisic', 'Train')
        test_dir = os.path.join(self.data_dir, 'scisic', 'Test')
        # Load the full dataset for classes information
        full_dataset = ImageFolder(train_dir, transform=transform)
        classes = full_dataset.classes  # Get classes from the full dataset
        # Load train dataset and split into train and validation
        train_dataset = ImageFolder(train_dir, transform=transform)
        train_dataset, val_dataset, _ = self.split_dataset(train_dataset)
        # Load test dataset separately
        test_dataset = ImageFolder(test_dir, transform=transform)
        return train_dataset, val_dataset, test_dataset, classes

    # Dataset #4: tbcr
    def load_tbcr(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(self.data_dir, 'tbcr')
        full_dataset = ImageFolder(train_dir, transform=transform)
        train_dataset, val_dataset, test_dataset = self.split_dataset(full_dataset)
        return train_dataset, test_dataset, val_dataset, full_dataset.classes

    # Dataset #5: ccts
    def load_ccts(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Ensure this line is present
            # Add other transformations as necessary
        ])
        train_dir = os.path.join(self.data_dir, 'ccts', 'train')
        test_dir = os.path.join(self.data_dir, 'ccts', 'test')
        valid_dir = os.path.join(self.data_dir, 'ccts', 'valid')
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
        valid_dataset = ImageFolder(valid_dir, transform=transform)
        return train_dataset, test_dataset, valid_dataset, train_dataset.classes

    @staticmethod
    def split_dataset(dataset):
        train_size = int(0.7 * len(dataset))  # 70% of the dataset for training
        val_size = int(0.15 * len(dataset))  # 15% of the dataset for validation
        test_size = len(dataset) - train_size - val_size  # 15% of the dataset for testing
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset

    def get_and_print_classes(self):
        *_, classes = self.load()
        if classes is not None:
            if isinstance(classes, list):
                logging.info(f"Classes for dataset: {classes}")
            else:
                logging.info("Classes for dataset: The classes attribute is not a list.")
            return classes
        else:
            raise ValueError("No classes found for dataset.")

    def get_input_channels(self):
        loaded_datasets = self.load()  # Renamed variable
        train_dataset = loaded_datasets[0]
        sample_img, _ = train_dataset[0]
        return sample_img.shape[0]

    def print_class_counts(self):
        logging.info("Calling print_class_counts method")
        train_dir = os.path.join(self.data_dir, self.dataset_name, 'Train')
        if not os.path.exists(train_dir):
            data_dir = os.path.join(self.data_dir, self.dataset_name)
        else:
            data_dir = train_dir
        dataset = ImageFolder(data_dir)
        class_counts = [0] * len(dataset.classes)
        for _, label in dataset:
            class_counts[label] += 1
        logging.info(f"All classes: {dataset.classes}")
        logging.info(f"All class counts: {class_counts}")

    class CustomImageDataset(Dataset):
        def __init__(self, root_dir=None, transform=None, data=None, length=None):
            self.transform = transform
            if root_dir is not None:
                self.root_dir = root_dir
                self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
                self.labels = [int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(root_dir)]
            elif data is not None:
                self.image_data = data
                self.length = length
            else:
                raise ValueError("Either root_dir or data must be provided.")

        def __len__(self):
            return len(self.image_paths) if hasattr(self, 'image_paths') else self.length

        def __getitem__(self, idx):
            if hasattr(self, 'root_dir'):
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert('RGB')
                label = self.labels[idx]
            else:
                image, label = self.image_data[idx]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

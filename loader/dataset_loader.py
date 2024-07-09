# dataset_loader.py

import os
import logging

import numpy
import numpy as np
import pandas as pd
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
            # 'miccai_brats2020': self.load_miccai_brats2020,
            # 'ccts': self.load_ccts,
            # 'tbcr': self.load_tbcr,
            'scisic': self.load_scisic,

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
            # 'miccai_brats2020': DatasetLoader('miccai_brats2020', data_dir),
            # 'ccts': DatasetLoader('ccts', data_dir),
            # 'tbcr': DatasetLoader('tbcr', data_dir),
            'scisic': DatasetLoader('scisic', data_dir),
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

    def load_miccai_brats2020(self):
        train_dir = os.path.join(self.data_dir, 'miccai_brats2020', 'MICCAI_BraTS2020_TrainingData')
        val_dir = os.path.join(self.data_dir, 'miccai_brats2020', 'MICCAI_BraTS2020_ValidationData')
        name_mapping = os.path.join(train_dir, 'name_mapping.csv')
        survival_info = os.path.join(train_dir, 'survival_info.csv')
        name_mapping_val = os.path.join(val_dir, 'name_mapping_validation_data.csv')
        survival_eval = os.path.join(val_dir, 'survival_evaluation.csv')

        train_dataset = self.CustomImageDataset(train_dir, name_mapping, survival_info)
        val_dataset = self.CustomImageDataset(val_dir, name_mapping_val, survival_eval, validation=True)

        # Check if val_dataset and test_dataset exist
        if val_dataset.__len__() > 0:
            train_dataset = train_dataset + val_dataset
            return train_dataset, None, None, None

        train_dataset, val_split_dataset, test_dataset = self.split_dataset(train_dataset)
        return train_dataset, val_split_dataset, test_dataset, None

    @staticmethod
    def split_dataset(dataset):
        if dataset is None:
            return None, None, None

        if isinstance(dataset, tuple):
            train_dataset = dataset[0]
            val_dataset = dataset[1]
            test_dataset = dataset[2]
        else:
            train_size = int(0.7 * len(dataset))  # 70% of the dataset for training
            val_size = int(0.15 * len(dataset))  # 15% of the dataset for validation
            test_size = len(dataset) - train_size - val_size  # Remaining for testing

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
        def __init__(self, root_dir, name_mapping_file, survival_info_file, transform=None, validation=False):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.load_data(root_dir, name_mapping_file, survival_info_file, validation)

        def load_data(self, root_dir, name_mapping_file, survival_info_file, validation):
            name_mapping = pd.read_csv(name_mapping_file)
            survival_info = pd.read_csv(survival_info_file)

            for _, row in name_mapping.iterrows():
                subject_id = row['BraTS_2020_subject_ID']
                subject_dir = os.path.join(root_dir, str(subject_id))
                for file in os.listdir(subject_dir):
                    if file.endswith('.nii'):
                        self.image_paths.append(os.path.join(subject_dir, file))
                        if 'seg' in file:
                            self.labels.append(subject_id)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = nib.load(image_path).get_fdata()
            image = np.expand_dims(image, axis=0)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

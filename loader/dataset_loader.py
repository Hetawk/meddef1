# dataset_loader.py

import os
import logging

import numpy
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import nibabel as nib
from typing import Tuple, Optional, Union, cast
from torch.utils.data import Dataset, DataLoader, random_split
from collections.abc import Sized


class DatasetLoader:
    """
    A class to manage and load various datasets for machine learning tasks.

    This class provides an interface for loading several datasets by specifying
    the dataset name. Supported datasets include MNIST, Med, SCISIC, TBCR, CCTS,
    and MICCAI BraTS2020. It also supports custom datasets with the option of
    splitting into training, validation, and test sets.
        """

    def __init__(self, dataset_name, data_dir='./dataset'):
        """
       Initializes the DatasetLoader with the given dataset name and directory.

       Args:
           dataset_name (str): The name of the dataset to load.
           data_dir (str, optional): The directory where the dataset is located. Defaults to './dataset'.

       Raises:
           ValueError: If the dataset_name is not recognized or not available in datasets_dict.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.datasets_dict = {
            # 'mnist': self.load_mnist,
            # 'med': self.load_med,
            # 'miccai_brats2020': self.load_miccai_brats2020,
            # 'ccts': self.load_ccts,
            # 'tbcr': self.load_tbcr,
            # 'scisic': self.load_scisic,
            'rotc': self.load_rotc,

            # Add more datasets here...
        }
        if self.dataset_name not in self.datasets_dict:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")
        logging.info(f"DatasetLoader initialized for {dataset_name}.")

    @staticmethod
    def get_all_datasets(data_dir='./dataset'):
        """
        Returns a dictionary of all available datasets with their corresponding DatasetLoader instances.

        Args:
            data_dir (str, optional): The directory where the datasets are located. Defaults to './dataset'.

        Returns:
            dict: A dictionary where keys are dataset names and values are DatasetLoader instances.
        """
        return {
            # 'mnist': DatasetLoader('mnist', data_dir),
            # 'med': DatasetLoader('med', data_dir),
            # 'miccai_brats2020': DatasetLoader('miccai_brats2020', data_dir),
            # 'ccts': DatasetLoader('ccts', data_dir),
            # 'tbcr': DatasetLoader('tbcr', data_dir),
            # 'scisic': DatasetLoader('scisic', data_dir),
            'rotc': DatasetLoader('rotc', data_dir),
        }

    def load(self):
        """
        Loads the dataset specified by `dataset_name` and returns the training, validation, and test DataLoaders.

        Returns:
            tuple: A tuple containing:
                - train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
                - val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
                - test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.

        Raises:
            ValueError: If the dataset_name is not recognized or not available in datasets_dict.
        """
        logging.info(f"Loading dataset: {self.dataset_name}.")
        try:
            train_loader, val_loader, test_loader, _ = self.datasets_dict[self.dataset_name]()
            return train_loader, val_loader, test_loader
        except KeyError:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")

    # Dataset #1: MNIST
    def load_mnist(self):
        """
        Loads the MNIST dataset and returns DataLoaders for the training, validation, and test datasets.

        Returns:
            tuple: A tuple containing:
                - train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
                - val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
                - test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
                - classes (list): List of class labels.
        """
        mnist_train = datasets.MNIST(os.path.join(self.data_dir, 'mnist'), train=True, download=True)
        mnist_test = datasets.MNIST(os.path.join(self.data_dir, 'mnist'), train=False, download=True)
        train_dataset, val_dataset, _ = self.split_dataset(mnist_train)

        return train_dataset, val_dataset, mnist_test, mnist_train.classes

    # Dataset #3: scisic
    def load_scisic(self):
        """
        Loads the SCISIC dataset and returns DataLoaders for the training, validation, and test datasets.


        Returns:
            tuple: A tuple containing:
                - train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
                - val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
                - test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
                - classes (list): List of class labels.
        """
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
        """
       Loads the TBCR dataset and returns DataLoaders for the training, validation, and test datasets.


        Returns:
            tuple: A tuple containing:
                - train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
                - val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
                - test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
                - classes (list): List of class labels.
        """
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
        """
        Chest CT-Scan images Dataset
        https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

        Loads the CCTS dataset and returns the training, validation, and test datasets.
        Returns:
            tuple: A tuple containing:
                - train_dataset (torch.utils.data.Dataset): The training dataset.
                - test_dataset (torch.utils.data.Dataset): The test dataset.
                - val_dataset (torch.utils.data.Dataset): The validation dataset.
                - classes (list): List of class labels.
        """
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

    # Dataset #6: rotc
    def load_rotc(self):
        """
        Loads the ROTC dataset and returns DataLoaders for the training, validation, and test datasets.


        Returns:
            tuple: A tuple containing:
                - train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
                - val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
                - test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
                - classes (list): List of class labels.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(self.data_dir, 'rotc', 'train')
        val_dir = os.path.join(self.data_dir, 'rotc', 'val')
        test_dir = os.path.join(self.data_dir, 'rotc', 'test')

        train_dataset = ImageFolder(train_dir, transform=transform)
        val_dataset = ImageFolder(val_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)

        return train_dataset, val_dataset, test_dataset, train_dataset.classes

    def load_miccai_brats2020(self):
        """
        Loads the MICCAI BraTS2020 dataset and returns the training, validation, and test datasets.

        Returns:
            tuple: A tuple containing:
                - train_dataset (torch.utils.data.Dataset): The combined training dataset (including validation if available).
                - val_dataset (torch.utils.data.Dataset or None): The validation dataset, or None if not available.
                - test_dataset (torch.utils.data.Dataset or None): The test dataset, or None if not available.
                - classes (None): This dataset does not use class labels in this implementation.
        """
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
    def split_dataset(dataset: Union[Dataset, Tuple[Dataset, ...], Sized]) -> Tuple[
        Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """
        Splits the dataset into training, validation, and test sets based on specified proportions.
        If the dataset cannot be split (e.g., it is None or empty), returns a tuple of (None, None, None).

        Args:
            dataset (torch.utils.data.Dataset): The dataset to split.

        Returns:
            tuple: A tuple containing:
                - train_dataset (torch.utils.data.Dataset or None): The training dataset.
                - val_dataset (torch.utils.data.Dataset or None): The validation dataset.
                - test_dataset (torch.utils.data.Dataset or None): The test dataset.
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

    def get_input_channels(self):
        """
        Retrieves the number of input channels from the training dataset.

        Returns:
            int: The number of input channels in the dataset.
        """
        loaded_datasets = self.load()  # Renamed variable
        train_dataset = loaded_datasets[0]
        sample_img, _ = train_dataset[0]
        return sample_img.shape[0]

    def print_class_counts(self):
        """
        Logs the count of samples per class in the dataset.
        """
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
        """
       A custom dataset class for loading images from a specified directory, with associated labels.

       Attributes:
           root_dir (str): The root directory containing the images.
           transform (callable, optional): A function/transform to apply to the images.
           image_paths (list): List of paths to the image files.
           labels (list): List of labels corresponding to the images.

       Methods:
           __init__(root_dir, name_mapping_file, survival_info_file, transform=None, validation=False):
               Initializes the dataset with the specified root directory and files.
           load_data(root_dir, name_mapping_file, survival_info_file, validation):
               Loads image paths and labels from the provided files.
           __len__():
               Returns the number of images in the dataset.
           __getitem__(idx):
               Retrieves an image and its label by index.
       """

        def __init__(self, root_dir, name_mapping_file, survival_info_file, transform=None, validation=False):
            """
            Initializes the CustomImageDataset with the specified root directory and files.

            Args:
                root_dir (str): The root directory containing the images.
                name_mapping_file (str): Path to the CSV file mapping image names to labels.
                survival_info_file (str): Path to the CSV file containing survival information.
                transform (callable, optional): A function/transform to apply to the images. Defaults to None.
                validation (bool, optional): Indicates if the dataset is for validation. Defaults to False.
            """
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.load_data(root_dir, name_mapping_file, survival_info_file, validation)

        def load_data(self, root_dir, name_mapping_file, survival_info_file, validation):
            """
            Loads image paths and labels from the provided files.

            Args:
                root_dir (str): The root directory containing the images.
                name_mapping_file (str): Path to the CSV file mapping image names to labels.
                survival_info_file (str): Path to the CSV file containing survival information.
                validation (bool): Indicates if the dataset is for validation.
            """
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
            """
            Returns the number of images in the dataset.

            Returns:
                int: The number of images in the dataset.
            """
            return len(self.image_paths)

        def __getitem__(self, idx):
            """
           Retrieves an image and its label by index.

           Args:
               idx (int): The index of the image to retrieve.

           Returns:
               tuple: A tuple containing:
                   - image (numpy.ndarray): The image data.
                   - label (str): The label associated with the image.
           """
            image_path = self.image_paths[idx]
            image = nib.load(image_path).get_fdata()
            image = np.expand_dims(image, axis=0)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

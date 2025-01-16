# dataset_loader.py

import os
import logging

import numpy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import nibabel as nib
from typing import Tuple, Optional, Union, cast
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from collections.abc import Sized

class DatasetLoader:
    """
    A class to manage and load various datasets for machine learning tasks.

    This class provides an interface for loading several datasets by specifying
    the dataset name. Supported datasets include MNIST, Med, SCISIC, TBCR, CCTS,
    and MICCAI BraTS2020. It also supports custom datasets with the option of
    splitting into training, validation, and test sets.
        """
    _dataset_instance = None

    pin_memory = True

    def __new__(cls, dataset_name, data_dir='./dataset'):
        if cls._dataset_instance is None:
            cls._dataset_instance = super(DatasetLoader, cls).__new__(cls)
            cls._dataset_instance._initialized = False
        return cls._dataset_instance

    def __init__(self, dataset_name, data_dir='./dataset'):
        """
       Initializes the DatasetLoader with the given dataset name and directory.

       Args:
           dataset_name (str): The name of the dataset to load.
           data_dir (str, optional): The directory where the dataset is located. Defaults to './dataset'.

       Raises:
           ValueError: If the dataset_name is not recognized or not available in datasets_dict.
        """
        if self._initialized:
            return
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.datasets_dict = {
            # 'mnist': self.load_mnist,
            # 'med': self.load_med,
            # 'miccai_brats2020': self.load_miccai_brats2020,
            'ccts': self.load_ccts,
            # 'tbcr': self.load_tbcr,
            'scisic': self.load_scisic,
            'rotc': self.load_rotc,
            'kvasir': self.load_kvasir,
            'dermnet': self.load_dermnet,
            'chest_xray': self.load_chest_xray

            # Add more datasets here...
        }
        if self.dataset_name not in self.datasets_dict:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")
        logging.info(f"DatasetLoader initialized for {dataset_name}.")
        DatasetLoader._dataset_instance = self
        self._initialized = True


    @staticmethod
    def get_all_datasets(dataset_names, data_dir='./dataset'):
        """
        Returns a dictionary of all available datasets with their corresponding DatasetLoader instances.

        Args:
            data_dir (str, optional): The directory where the datasets are located. Defaults to './dataset'.

        Returns:
            dict: A dictionary where keys are dataset names and values are DatasetLoader instances.
            :param data_dir:
            :param dataset_names:
        """
        datasets_dict = {}
        for dataset_name in dataset_names:
            datasets_dict[dataset_name] = DatasetLoader(dataset_name, data_dir)
        return datasets_dict

    def load(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
        Loads the dataset specified by `dataset_name` and returns the training, validation, and test DataLoaders.

        Args:
            train_batch_size (int): Batch size for the training DataLoader.
            val_batch_size (int): Batch size for the validation DataLoader.
            test_batch_size (int): Batch size for the test DataLoader.
            num_workers (int): Number of worker processes for data loading.
            pin_memory (bool): Whether to use pinned memory for data loading.

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
            return self.datasets_dict[self.dataset_name](train_batch_size, val_batch_size, test_batch_size, num_workers,
                                                         pin_memory)
        except KeyError:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")

    def check_for_corrupted_images(self, directory, transform):
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
    # Dataset #3: scisic
    def load_scisic(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
        Skin Cancer ISIC
        https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
        Loads the SCISIC dataset and returns DataLoaders for the training, validation, and test datasets.

        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_dir = os.path.join(self.data_dir, 'scisic', 'Train')
        test_dir = os.path.join(self.data_dir, 'scisic', 'Test')

        # Check for corrupted images in the training directory
        self.check_for_corrupted_images(train_dir, data_transforms['train'])
        self.check_for_corrupted_images(test_dir, data_transforms['test'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, _ = self.split_dataset(full_dataset)
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = self.get_WeightedRandom_Sampler(train_dataset, full_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def load_kvasir(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
        Kvasir Dataset for Classification and Segmentation
        https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation
        Loads the Kvasir dataset and returns DataLoaders for the training, validation, and test datasets.
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_dir = os.path.join(self.data_dir, 'kvasir', 'train')

        # Check for corrupted images in the training directory
        self.check_for_corrupted_images(train_dir, data_transforms['train'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, test_dataset = self.split_dataset(full_dataset)

        weight_sampler = self.get_WeightedRandom_Sampler(train_dataset, full_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def load_dermnet(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
        Dermnet
        https://www.kaggle.com/datasets/shubhamgoel27/dermnet
        Loads the Dermnet dataset and returns DataLoaders for the training, validation, and test datasets.

        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_dir = os.path.join(self.data_dir, 'dermnet', 'train')
        test_dir = os.path.join(self.data_dir, 'dermnet', 'test')

        # Check for corrupted images in the training directory
        self.check_for_corrupted_images(train_dir, data_transforms['train'])
        self.check_for_corrupted_images(test_dir, data_transforms['test'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, _ = self.split_dataset(full_dataset)
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = self.get_WeightedRandom_Sampler(train_dataset, full_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

        return train_loader, val_loader, test_loader
    # # Dataset #4: tbcr
    # def load_tbcr(self):
    #     """
    #     Tuberculosis (TB) Chest X-ray Database
    #     https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
    #    Loads the TBCR dataset and returns DataLoaders for the training, validation, and test datasets.
    #
    #     """
    #     transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor()
    #     ])
    #     train_dir = os.path.join(self.data_dir, 'tbcr')
    #     full_dataset = ImageFolder(train_dir, transform=transform)
    #     train_dataset, val_dataset, test_dataset = self.split_dataset(full_dataset)
    #
    #     return train_dataset, test_dataset, val_dataset, full_dataset.classes

    # Dataset #5: ccts
    def load_ccts(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
                Chest CT-Scan images Dataset
                https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
                Data: 613 images -> Batch size to use: 8 to 16  -> Test set batch can more than training
                Works best: 4 - 8
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_dir = os.path.join(self.data_dir, 'ccts', 'train')
        test_dir = os.path.join(self.data_dir, 'ccts', 'test')
        valid_dir = os.path.join(self.data_dir, 'ccts', 'valid')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])
        val_dataset = ImageFolder(valid_dir, data_transforms['val'])

        weight_sampler = self.get_WeightedRandom_Sampler(train_dataset, train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def load_chest_xray(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
                Chest CT-Scan images Dataset
                https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_dir = os.path.join(self.data_dir, 'chest_xray', 'train')
        test_dir = os.path.join(self.data_dir, 'chest_xray', 'test')
        valid_dir = os.path.join(self.data_dir, 'chest_xray', 'val')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])
        val_dataset = ImageFolder(valid_dir, data_transforms['val'])

        weight_sampler = self.get_WeightedRandom_Sampler(train_dataset, train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    # Dataset #6: rotc
    def load_rotc(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
        Retinal OCT Images (optical coherence tomography)
        https://www.kaggle.com/datasets/paultimothymooney/kermany2018
        Loads the ROTC dataset and returns DataLoaders for the training, validation, and test datasets.

        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_dir = os.path.join(self.data_dir, 'rotc', 'train')
        val_dir = os.path.join(self.data_dir, 'rotc', 'val')
        test_dir = os.path.join(self.data_dir, 'rotc', 'test')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        val_dataset = ImageFolder(val_dir, data_transforms['val'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = self.get_WeightedRandom_Sampler(train_dataset, train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def load_miccai_brats2020(self):
        """
        Loads the MICCAI BraTS2020 dataset and returns the training, validation, and test datasets.

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

    @staticmethod
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

    @staticmethod
    def get_dataloader_target_class_number(dataLoader):
        if DatasetLoader._dataset_instance is None:
            raise ValueError("DatasetLoader instance is not initialized.")

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

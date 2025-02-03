# dataset_loader.py

import os
import logging

import numpy
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
import nibabel as nib
from torch.utils.data import Dataset
from loader.preprocessing import get_default_transforms, preprocess_dataset, get_WeightedRandom_Sampler, split_dataset, \
    check_for_corrupted_images


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

    # Dataset #3: scisic
    def load_scisic(self, train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory):
        """
        Skin Cancer ISIC: 2358
        https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
        Loads the SCISIC dataset and returns DataLoaders for the training, validation, and test datasets.

        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'scisic', 'Train')
        test_dir = os.path.join(self.data_dir, 'scisic', 'Test')

        # Check for corrupted images in the training directory
        check_for_corrupted_images(train_dir, data_transforms['train'])
        check_for_corrupted_images(test_dir, data_transforms['test'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, full_dataset)
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
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'kvasir', 'train')

        # Check for corrupted images in the training directory
        check_for_corrupted_images(train_dir, data_transforms['train'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, full_dataset)
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
        19,500 images ->
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'dermnet', 'train')
        test_dir = os.path.join(self.data_dir, 'dermnet', 'test')

        # Check for corrupted images in the training directory
        check_for_corrupted_images(train_dir, data_transforms['train'])
        check_for_corrupted_images(test_dir, data_transforms['test'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, full_dataset)
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
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'ccts', 'train')
        test_dir = os.path.join(self.data_dir, 'ccts', 'test')
        valid_dir = os.path.join(self.data_dir, 'ccts', 'valid')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])
        val_dataset = ImageFolder(valid_dir, data_transforms['val'])

        # # Preprocess the train_dataset
        # train_dataset_cleaned = preprocess_dataset(train_dataset)
        # # Convert the cleaned dataset back to ImageFolder format
        # train_dataset = self.CustomImageDataset(train_dataset_cleaned, transform=data_transforms['train'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, train_dataset)
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
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'chest_xray', 'train')
        test_dir = os.path.join(self.data_dir, 'chest_xray', 'test')
        valid_dir = os.path.join(self.data_dir, 'chest_xray', 'val')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])
        val_dataset = ImageFolder(valid_dir, data_transforms['val'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, train_dataset)
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
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'rotc', 'train')
        val_dir = os.path.join(self.data_dir, 'rotc', 'val')
        test_dir = os.path.join(self.data_dir, 'rotc', 'test')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        val_dataset = ImageFolder(val_dir, data_transforms['val'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, train_dataset)
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





    class CustomImageDataset(Dataset):

        def __init__(self, root_dir, name_mapping_file, survival_info_file, transform=None, validation=False,
                     file_extension='.nii', label_extraction_func=None):

            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.file_extension = file_extension
            self.label_extraction_func = label_extraction_func
            self.load_data(root_dir, name_mapping_file, survival_info_file, validation)

        def load_data(self, root_dir, name_mapping_file, survival_info_file, validation):
            name_mapping = pd.read_csv(name_mapping_file)
            survival_info = pd.read_csv(survival_info_file)

            for _, row in name_mapping.iterrows():
                subject_id = row['BraTS_2020_subject_ID']
                subject_dir = os.path.join(root_dir, str(subject_id))
                for file in os.listdir(subject_dir):
                    if file.endswith(self.file_extension):
                        self.image_paths.append(os.path.join(subject_dir, file))
                        if self.label_extraction_func:
                            self.labels.append(self.label_extraction_func(file))
                        else:
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

# dataset_loader.py

import os
import logging
from typing import Tuple, Dict, List, Callable

import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# Assume these functions are defined in preprocessing.py
from loader.preprocessing import (
    get_default_transforms,
    preprocess_dataset,
    get_WeightedRandom_Sampler,
    split_dataset,
    check_for_corrupted_images
)

# Define a type alias for dataset-loading callables:
DatasetLoaderCallable = Callable[
    [int, int, int, int, bool],
    Tuple[DataLoader, DataLoader, DataLoader]
]


class DatasetLoader:
    """
    A class to manage and load various datasets for machine learning tasks.

    Supported datasets include: 'ccts', 'scisic', 'rotc', 'kvasir', 'dermnet', 'chest_xray'.
    """

    _dataset_instance = None
    # pin_memory = True

    def __new__(cls, dataset_name: str, data_dir: str = './dataset'):
        if cls._dataset_instance is None:
            cls._dataset_instance = super(DatasetLoader, cls).__new__(cls)
            cls._dataset_instance._initialized = False
        return cls._dataset_instance

    def __init__(self, dataset_name: str, data_dir: str = './dataset'):
        """
        Initializes the DatasetLoader with the given dataset name and directory.

        Args:
            dataset_name (str): The name of the dataset to load.
            data_dir (str, optional): The directory where the dataset is located. Defaults to './dataset'.

        Raises:
            ValueError: If the dataset_name is not recognized.
        """
        if self._initialized:
            return

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.pin_memory = False
        self.datasets_dict: Dict[str, DatasetLoaderCallable] = {
            'ccts': self.load_ccts,
            'scisic': self.load_scisic,
            'rotc': self.load_rotc,
            'kvasir': self.load_kvasir,
            'dermnet': self.load_dermnet,
            'chest_xray': self.load_chest_xray
            # Add more datasets here if needed.
        }
        if self.dataset_name not in self.datasets_dict:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")
        logging.info(f"DatasetLoader initialized for {dataset_name}.")
        DatasetLoader._dataset_instance = self
        self._initialized = True

    @staticmethod
    def get_all_datasets(dataset_names: List[str], data_dir: str = './dataset') -> Dict[str, "DatasetLoader"]:
        """
        Returns a dictionary of DatasetLoader instances for all provided dataset names.

        Args:
            dataset_names (List[str]): List of dataset names.
            data_dir (str, optional): Directory where the datasets are located.

        Returns:
            Dict[str, DatasetLoader]: Dictionary mapping dataset name to DatasetLoader instance.
        """
        datasets_dict: Dict[str, DatasetLoader] = {}
        for dataset_name in dataset_names:
            datasets_dict[dataset_name] = DatasetLoader(dataset_name, data_dir)
        return datasets_dict

    def load(self,
             train_batch_size: int,
             val_batch_size: int,
             test_batch_size: int,
             num_workers: int,
             pin_memory: bool
             ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the specified dataset and returns the training, validation, and test DataLoaders.

        Args:
            train_batch_size (int): Batch size for the training DataLoader.
            val_batch_size (int): Batch size for the validation DataLoader.
            test_batch_size (int): Batch size for the test DataLoader.
            num_workers (int): Number of worker processes for data loading.
            pin_memory (bool): Whether to use pinned memory.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
        """
        logging.info(f"Loading dataset: {self.dataset_name}.")
        return self.datasets_dict[self.dataset_name](
            train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory
        )

    def load_scisic(self,
                    train_batch_size: int,
                    val_batch_size: int,
                    test_batch_size: int,
                    num_workers: int,
                    pin_memory: bool
                    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the SCISIC dataset and returns DataLoaders for training, validation, and test sets.
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'scisic', 'Train')
        test_dir = os.path.join(self.data_dir, 'scisic', 'Test')

        # Check for corrupted images
        check_for_corrupted_images(train_dir, data_transforms['train'])
        check_for_corrupted_images(test_dir, data_transforms['test'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, full_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def load_kvasir(self,
                    train_batch_size: int,
                    val_batch_size: int,
                    test_batch_size: int,
                    num_workers: int,
                    pin_memory: bool
                    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the Kvasir dataset and returns DataLoaders for training, validation, and test sets.
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'kvasir', 'train')
        check_for_corrupted_images(train_dir, data_transforms['train'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, full_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def load_dermnet(self,
                     train_batch_size: int,
                     val_batch_size: int,
                     test_batch_size: int,
                     num_workers: int,
                     pin_memory: bool
                     ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the Dermnet dataset and returns DataLoaders for training, validation, and test sets.
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'dermnet', 'train')
        test_dir = os.path.join(self.data_dir, 'dermnet', 'test')

        check_for_corrupted_images(train_dir, data_transforms['train'])
        check_for_corrupted_images(test_dir, data_transforms['test'])

        full_dataset = ImageFolder(train_dir, data_transforms['train'])
        train_dataset, val_dataset, _ = split_dataset(full_dataset)
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, full_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def load_ccts(self,
                  train_batch_size: int,
                  val_batch_size: int,
                  test_batch_size: int,
                  num_workers: int,
                  pin_memory: bool
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the Chest CT-Scan images Dataset (CCTS) and returns DataLoaders.
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'ccts', 'train')
        test_dir = os.path.join(self.data_dir, 'ccts', 'test')
        valid_dir = os.path.join(self.data_dir, 'ccts', 'valid')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])
        val_dataset = ImageFolder(valid_dir, data_transforms['val'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def load_chest_xray(self,
                        train_batch_size: int,
                        val_batch_size: int,
                        test_batch_size: int,
                        num_workers: int,
                        pin_memory: bool
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the Chest X-ray dataset and returns DataLoaders for training, validation, and test sets.
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'chest_xray', 'train')
        test_dir = os.path.join(self.data_dir, 'chest_xray', 'test')
        valid_dir = os.path.join(self.data_dir, 'chest_xray', 'val')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])
        val_dataset = ImageFolder(valid_dir, data_transforms['val'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def load_rotc(self,
                  train_batch_size: int,
                  val_batch_size: int,
                  test_batch_size: int,
                  num_workers: int,
                  pin_memory: bool
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the Retinal OCT Images (ROTC) dataset and returns DataLoaders.
        """
        data_transforms = get_default_transforms()

        train_dir = os.path.join(self.data_dir, 'rotc', 'train')
        val_dir = os.path.join(self.data_dir, 'rotc', 'val')
        test_dir = os.path.join(self.data_dir, 'rotc', 'test')

        train_dataset = ImageFolder(train_dir, data_transforms['train'])
        val_dataset = ImageFolder(val_dir, data_transforms['val'])
        test_dataset = ImageFolder(test_dir, data_transforms['test'])

        weight_sampler = get_WeightedRandom_Sampler(train_dataset, train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=weight_sampler, batch_size=train_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def get_input_channels(self,
                           train_batch_size: int,
                           val_batch_size: int,
                           test_batch_size: int,
                           num_workers: int,
                           pin_memory: bool
                           ) -> int:
        """
        Retrieves the number of input channels from the training dataset.
        """
        train_loader, _, _ = self.load(train_batch_size, val_batch_size, test_batch_size, num_workers, pin_memory)
        train_dataset = train_loader.dataset
        sample_img, _ = train_dataset[0]
        return sample_img.shape[0]

    class CustomImageDataset(Dataset):
        def __init__(self,
                     root_dir: str,
                     name_mapping_file: str,
                     survival_info_file: str,
                     transform=None,
                     validation: bool = False,
                     file_extension: str = '.nii',
                     label_extraction_func=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths: List[str] = []
            self.labels: List = []
            self.file_extension = file_extension
            self.label_extraction_func = label_extraction_func
            self.load_data(root_dir, name_mapping_file, survival_info_file, validation)

        def load_data(self, root_dir: str, name_mapping_file: str, survival_info_file: str, validation: bool):
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

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int):
            image_path = self.image_paths[idx]
            image = nib.load(image_path).get_fdata()
            image = np.expand_dims(image, axis=0)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

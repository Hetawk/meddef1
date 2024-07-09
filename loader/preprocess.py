# preprocess.py

import logging
import os
import torch
from torchvision import transforms
from loader.dataset_loader import DatasetLoader
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter


class Preprocessor:
    def __init__(self, model_type, dataset_name, task_name, data_dir='./dataset', hyperparams=None):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.data_dir = data_dir
        self.hyperparams = hyperparams or {}
        logging.info(f"Preprocessor initialized with model type {model_type}.")

    def wrap_datasets_in_dataloaders(self, train_dataset, val_dataset=None, test_dataset=None, shuffle=True):
        batch_size = self.hyperparams['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False) if test_dataset is not None else None
        return train_loader, val_loader, test_loader

    def preprocess(self, train_dataset, val_dataset=None, test_dataset=None, input_channels=None):
        logging.info(f"Preprocessing data for {self.model_type} model with {input_channels} input channels.")
        self.summarize_dataset(train_dataset, val_dataset, test_dataset)
        self.print_classes()
        transform = self.get_transforms(input_channels, train_dataset)
        if train_dataset is not None:
            train_dataset.transform = transform
        if val_dataset is not None:
            val_dataset.transform = transform
        if test_dataset is not None:
            test_dataset.transform = transform
        train_dataset = self.verify_labels(train_dataset)
        if val_dataset is not None:
            val_dataset = self.verify_labels(val_dataset)
        if test_dataset is not None:
            test_dataset = self.verify_labels(test_dataset)
        return train_dataset, val_dataset, test_dataset

    def print_classes(self):
        dataset_loader = DatasetLoader(self.dataset_name, self.data_dir)
        try:
            dataset_loader.get_and_print_classes()
            dataset_loader.print_class_counts()
        except ValueError as e:
            logging.error(e)
            raise

    def get_transforms(self, input_channels, train_dataset=None):
        pil_transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            self.de_texturize_transform(),
            self.de_colorize_transform(),
            self.edge_enhance_transform(),
        ]
        to_tensor_transform = transforms.ToTensor()
        tensor_transform_list = [
            self.salient_edge_map_transform(),
        ]
        all_transforms = pil_transform_list + [to_tensor_transform] + tensor_transform_list
        if input_channels is not None:
            self.add_grayscale_transform(all_transforms, input_channels, train_dataset)
            self.add_normalization_transform(all_transforms, input_channels)
        return transforms.Compose(all_transforms)

    @staticmethod
    def de_texturize_transform():
        return transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)))

    @staticmethod
    def de_colorize_transform():
        return transforms.Grayscale(num_output_channels=1)

    @staticmethod
    def edge_enhance_transform():
        return transforms.Lambda(lambda img: img.filter(ImageFilter.EDGE_ENHANCE))

    @staticmethod
    def salient_edge_map_transform():
        def edge_detection(tensor_img):
            if tensor_img.shape[0] == 3:
                img = tensor_img.numpy().transpose(1, 2, 0).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = tensor_img.numpy().squeeze(0).astype(np.uint8)
            edges = cv2.Canny(img, 100, 200)
            edges = np.stack([edges, edges, edges], axis=0)
            return torch.tensor(edges, dtype=torch.float32) / 255.0

        return transforms.Lambda(edge_detection)

    def add_grayscale_transform(self, transform_list, input_channels, train_dataset=None):
        if input_channels == 3 and train_dataset is not None and hasattr(train_dataset, 'classes') and \
                train_dataset.classes[0] == '0':
            transform_list.insert(1, transforms.Grayscale(num_output_channels=3))

    def add_normalization_transform(self, transform_list, input_channels):
        if input_channels == 1:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    @staticmethod
    def verify_labels(dataset):
        label_counts = Counter()
        for _, label in dataset:
            label_counts[label] += 1
        majority_label = label_counts.most_common(1)[0][0]
        corrected_labels = [majority_label if label != majority_label else label for _, label in dataset]
        dataset.labels = corrected_labels
        return dataset

    def summarize_dataset(self, train_dataset, val_dataset=None, test_dataset=None):
        logging.info(f"Summarizing dataset: {self.dataset_name}")
        logging.info(f"Number of training samples: {len(train_dataset)}")
        if val_dataset is not None:
            logging.info(f"Number of validation samples: {len(val_dataset)}")
        if test_dataset is not None:
            logging.info(f"Number of test samples: {len(test_dataset)}")
        self.print_class_counts(train_dataset)
        self.calculate_basic_statistics(train_dataset)
        self.visualize_samples(self.model_type, train_dataset)

    @staticmethod
    def print_class_counts(dataset):
        logging.info("Calculating class counts")
        # Handle if the dataset is a Subset
        if isinstance(dataset, torch.utils.data.Subset):
            original_dataset = dataset.dataset
        else:
            original_dataset = dataset
        # Check if the 'classes' attribute exists
        if hasattr(original_dataset, 'classes'):
            classes = original_dataset.classes
        else:
            logging.warning("The dataset does not have a 'classes' attribute.")
            # Handle the case where 'classes' does not exist, e.g., by inferring classes or skipping
            return
        indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else range(len(original_dataset))
        # Initialize the class counts
        class_counts = [0] * len(classes)
        # Count the occurrences of each class in the subset
        for idx in indices:
            _, label = original_dataset[idx]
            class_counts[label] += 1
        logging.info(f"Classes: {classes}")
        logging.info(f"Class counts: {class_counts}")

    # @staticmethod
    # def calculate_basic_statistics(dataset):
    #     data_list = []
    #     for data, _ in dataset:
    #         if isinstance(data, np.ndarray):
    #             data_list.append(data)
    #         elif isinstance(data, np.memmap):
    #             data_list.append(data)
    #         else:
    #             raise TypeError(f"Unsupported data type encountered: {type(data)}")
    #
    #     data_array = np.array(data_list)
    #     mean = np.mean(data_array, axis=0)
    #     median = np.median(data_array, axis=0)
    #     std_dev = np.std(data_array, axis=0)
    #     global_mean = np.mean(mean)
    #     global_median = np.mean(median)
    #     global_std_dev = np.mean(std_dev)
    #     logging.info(
    #         f"Global Mean: {global_mean}, Global Median: {global_median}, Global Standard Deviation: {global_std_dev}")

    @staticmethod
    def calculate_basic_statistics(dataset):
        data_list = []
        for data, _ in dataset:
            data_list.append(data.numpy())
        data_array = np.array(data_list)
        mean = np.mean(data_array, axis=0)
        median = np.median(data_array, axis=0)
        std_dev = np.std(data_array, axis=0)
        global_mean = np.mean(mean)
        global_median = np.mean(median)
        global_std_dev = np.mean(std_dev)
        logging.info(
            f"Global Mean: {global_mean}, Global Median: {global_median}, Global Standard Deviation: {global_std_dev}")

    def visualize_samples(self, model_name, dataset, num_samples=5):
        output_dir = os.path.join('out', self.task_name, self.dataset_name, 'pre_visualization')
        os.makedirs(output_dir, exist_ok=True)

        # Accumulate all samples and labels
        sample_images = []
        sample_labels = []
        for i in range(num_samples):
            img, label = dataset[i]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
            if img.shape[2] == 1:  # Grayscale image
                img = img.squeeze()
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            sample_images.append(img)
            sample_labels.append(label)

        # Create a single figure for visualization
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            axs[i].imshow(sample_images[i])
            axs[i].set_title(f'Model: {model_name} | Label: {sample_labels[i]}')
            axs[i].axis('off')

        # Save the complete visualization
        output_path = os.path.join(output_dir, f'sample_visualization_model_{model_name}.png')
        plt.savefig(output_path)
        logging.info(f'Complete sample visualization saved to {output_path}')
        plt.close(fig)

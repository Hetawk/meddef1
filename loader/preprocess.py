# # preprocess.py
#
# import logging
# import os
# import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from torchvision.transforms import functional as f
# from PIL import Image, ImageFilter
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from collections import Counter
#
#
# class Preprocessor:
#     """
#         A class to preprocess datasets for training machine learning models, including
#         applying transformations, handling class imbalance, and visualizing samples.
#     """
#     def __init__(self, model_type, dataset_name, task_name, data_dir='./dataset', hyperparams=None):
#         """
#         Initializes the Preprocessor with the specified parameters.
#
#         Args:
#             model_type (str): The type of model being used.
#             dataset_name (str): The name of the dataset to preprocess.
#             task_name (str): The name of the task for which the dataset is being prepared.
#             data_dir (str, optional): The directory where the dataset is located. Defaults to './dataset'.
#             hyperparams (dict, optional): Hyperparameters for preprocessing, including batch size. Defaults to None.
#         """
#         self.model_type = model_type
#         self.dataset_name = dataset_name
#         self.task_name = task_name
#         self.data_dir = data_dir
#         self.hyperparams = hyperparams or {}
#         logging.info(f"Preprocessor initialized with model type {model_type}.")
#
#     def wrap_datasets_in_dataloaders(self, train_dataset, val_dataset=None, test_dataset=None, shuffle=True):
#         """
#         Wraps the provided datasets in DataLoader instances.
#
#         Args:
#             train_dataset (torch.utils.data.Dataset): The training dataset.
#             val_dataset (torch.utils.data.Dataset, optional): The validation dataset. Defaults to None.
#             test_dataset (torch.utils.data.Dataset, optional): The test dataset. Defaults to None.
#             shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
#
#         Returns:
#             tuple: A tuple containing:
#                 - train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
#                 - val_loader (torch.utils.data.DataLoader or None): The DataLoader for the validation dataset,
#                  or None if not provided.
#                 - test_loader (torch.utils.data.DataLoader or None): The DataLoader for the test dataset,
#                  or None if not provided.
#         """
#         batch_size = self.hyperparams['batch_size']
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
#         test_loader = DataLoader(test_dataset, batch_size=batch_size,
#                                  shuffle=False) if test_dataset is not None else None
#         return train_loader, val_loader, test_loader
#
#     def preprocess(self, train_dataset, val_dataset=None, test_dataset=None, input_channels=None):
#         """
#         Preprocesses the datasets by applying transformations and handling class imbalance.
#
#         Args:
#             train_dataset (torch.utils.data.Dataset): The training dataset.
#             val_dataset (torch.utils.data.Dataset, optional): The validation dataset. Defaults to None.
#             test_dataset (torch.utils.data.Dataset, optional): The test dataset. Defaults to None.
#             input_channels (int, optional): The number of input channels in the dataset. Defaults to None.
#
#         Returns:
#             tuple: A tuple containing:
#                 - train_dataset (torch.utils.data.Dataset): The preprocessed training dataset.
#                 - val_dataset (torch.utils.data.Dataset or None): The preprocessed validation dataset, or None if not provided.
#                 - test_dataset (torch.utils.data.Dataset or None): The preprocessed test dataset, or None if not provided.
#         """
#         logging.info(f"Preprocessing data for {self.model_type} model with {input_channels} input channels.")
#         # self.summarize_dataset(train_dataset, val_dataset, test_dataset)
#
#         transform = self.get_transforms(input_channels, train_dataset)
#         if train_dataset is not None:
#             train_dataset.transform = transform
#         if val_dataset is not None:
#             val_dataset.transform = transform
#         if test_dataset is not None:
#             test_dataset.transform = transform
#         train_dataset = self.verify_labels(train_dataset)
#         if val_dataset is not None:
#             val_dataset = self.verify_labels(val_dataset)
#         if test_dataset is not None:
#             test_dataset = self.verify_labels(test_dataset)
#         return train_dataset, val_dataset, test_dataset
#
#     def get_transforms(self, input_channels, train_dataset=None):
#         """
#         Generates a composition of transformations to apply to the dataset based on the dataset name.
#
#         Args:
#             input_channels (int): The number of input channels in the dataset.
#             train_dataset (torch.utils.data.Dataset, optional): The training dataset. Defaults to None.
#
#         Returns:
#             torchvision.transforms.Compose: The composed transformation to apply.
#         """
#         pil_transform_list = [
#
#             transforms.Resize((256, 256)),
#             # transforms.Resize((224, 224)),
#             transforms.CenterCrop((224, 224)),
#
#             # transforms.RandomHorizontalFlip(),
#
#             # transforms.RandomRotation(10),
#             # transforms.RandomResizedCrop(224),
#
#             # self.de_texturize_transform(),
#             # self.de_colorize_transform(),
#             # self.edge_enhance_transform(),
#         ]
#         if self.dataset_name == 'ccts':
#             pil_transform_list.extend([
#                 # self.de_texturize_transform(),
#                 # self.brightness_adjust_transform(1.5),
#                 # transforms.Grayscale(num_output_channels=1),
#             ])
#         elif self.dataset_name == 'tbcr':
#             pil_transform_list.extend([
#                 transforms.Grayscale(num_output_channels=1),
#                 transforms.Lambda(lambda img: self.handle_class_imbalance(img, train_dataset)),
#             ])
#             # if train_dataset:
#             #     weighted_sampler = self.get_weighted_sampler(train_dataset)
#             #     train_dataset = self.add_weighted_sampler(train_dataset, weighted_sampler)
#             #     return train_dataset # not sure what was done here, nothing was returned
#         elif self.dataset_name == 'scisic':
#             pil_transform_list.extend([
#                 # self.de_colorize_transform(),
#                 # self.edge_enhance_transform(),
#             ])
#         elif self.dataset_name == 'rotc':
#             pil_transform_list.extend([
#                 # self.de_texturize_transform(),
#                 # self.de_colorize_transform(),
#                 # self.edge_enhance_transform(),
#             ])
#
#         to_tensor_transform = transforms.ToTensor()
#         tensor_transform_list = [
#             # self.salient_edge_map_transform(),
#         ]
#         all_transforms = pil_transform_list + [to_tensor_transform] + tensor_transform_list
#
#         if input_channels is not None:
#             self.add_normalization_transform(all_transforms, input_channels)
#         return transforms.Compose(all_transforms)
#
#     @staticmethod
#     def de_texturize_transform():
#         """
#         Creates a transformation to apply Gaussian blur to images.
#
#         Returns:
#             torchvision.transforms.Lambda: The transformation that applies Gaussian blur.
#         """
#         return transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)))
#
#     @staticmethod
#     def de_colorize_transform():
#         """
#         Creates a transformation to convert images to grayscale.
#
#         Returns:
#             torchvision.transforms.Grayscale: The transformation that converts images to grayscale.
#         """
#         return transforms.Grayscale(num_output_channels=1)
#
#     @staticmethod
#     def edge_enhance_transform():
#         """
#         Creates a transformation to enhance edges in images.
#
#         Returns:
#             torchvision.transforms.Lambda: The transformation that enhances edges.
#         """
#         return transforms.Lambda(lambda img: img.filter(ImageFilter.EDGE_ENHANCE))
#
#     @staticmethod
#     def salient_edge_map_transform():
#         """
#         Creates a transformation to detect edges using the Canny edge detector.
#
#         Returns:
#             torchvision.transforms.Lambda: The transformation that applies Canny edge detection.
#         """
#         def edge_detection(tensor_img):
#             if tensor_img.shape[0] == 3:
#                 img = tensor_img.numpy().transpose(1, 2, 0).astype(np.uint8)
#                 img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             else:
#                 img = tensor_img.numpy().squeeze(0).astype(np.uint8)
#             edges = cv2.Canny(img, 100, 200)
#             edges = np.stack([edges, edges, edges], axis=0)
#             return torch.tensor(edges, dtype=torch.float32) / 255.0
#
#         return transforms.Lambda(edge_detection)
#
#     @staticmethod
#     def brightness_adjust_transform(factor):
#         """
#         Creates a transformation to adjust the brightness of images.
#
#         Args:
#             factor (float): The factor by which to adjust the brightness.
#
#         Returns:
#             torchvision.transforms.Lambda: The transformation that adjusts the brightness.
#         """
#         return transforms.Lambda(lambda img: f.adjust_brightness(img, factor))
#
#     def handle_class_imbalance(self, img, dataset):
#         """
#         Handles class imbalance by oversampling minority class images.
#
#         Args:
#             img (torch.Tensor): The image to process.
#             dataset (torch.utils.data.Dataset): The dataset containing class labels.
#
#         Returns:
#             torch.Tensor: The processed image.
#         """
#         # Oversample the minority class ('Tuberculosis') images in the 'tbcr' dataset
#         if hasattr(dataset, 'targets'):
#             targets = dataset.targets
#         elif hasattr(dataset, 'labels'):
#             targets = dataset.labels
#         else:
#             raise ValueError("Dataset must have 'targets' or 'labels' attribute.")
#
#         class_counts = np.bincount(targets)
#         majority_class = np.argmax(class_counts)
#         minority_class = 1 - majority_class  # Assuming minority class is 1 (Tuberculosis)
#
#         # If the image belongs to the minority class, perform oversampling
#         if targets[img] == minority_class:
#             img = Image.fromarray(img.numpy().astype(np.uint8))
#             img = self.augment_minority_class(img)
#             img = transforms.ToTensor()(img)
#
#         return img
#
#     def augment_minority_class(self, img):
#         """
#         Applies augmentation techniques to images of the minority class.
#
#         Args:
#             img (PIL.Image): The image to augment.
#
#         Returns:
#             PIL.Image: The augmented image.
#         """
#         # Define your oversampling technique here (e.g., rotation, flipping, etc.)
#         img = transforms.RandomRotation(10)(img)
#         return img
#
#     def add_weighted_sampler(self, dataset, sampler):
#         """
#         Adds a weighted sampler to the dataset for class imbalance handling.
#
#         Args:
#             dataset (torch.utils.data.Dataset): The dataset to modify.
#             sampler (torch.utils.data.WeightedRandomSampler): The sampler to add.
#
#         Returns:
#             torch.utils.data.Dataset: The dataset with the weighted sampler added.
#         """
#         dataset.targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
#         dataset.targets_sampler = sampler
#         return dataset
#
#     def get_weighted_sampler(self, dataset):
#         """
#         Creates a weighted sampler to handle class imbalance.
#
#         Args:
#             dataset (torch.utils.data.Dataset): The dataset to create the sampler for.
#
#         Returns:
#             torch.utils.data.WeightedRandomSampler: The weighted sampler.
#         """
#         targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
#         class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
#         weight = 1. / class_sample_count
#         samples_weight = np.array([weight[t] for t in targets])
#         sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
#         return sampler
#
#     def add_grayscale_transform(self, transform_list, input_channels, train_dataset=None):
#         """
#         Adds a grayscale transformation to the list of transformations if needed.
#
#         Args:
#             transform_list (list): The list of transformations to modify.
#             input_channels (int): The number of input channels in the dataset.
#             train_dataset (torch.utils.data.Dataset, optional): The training dataset. Defaults to None.
#         """
#         if input_channels == 3 and train_dataset is not None and hasattr(train_dataset, 'classes') and \
#                 train_dataset.classes[0] == '0':
#             transform_list.insert(1, transforms.Grayscale(num_output_channels=3))
#
#     def add_normalization_transform(self, transform_list, input_channels):
#         """
#         Adds normalization to the list of transformations based on input channels.
#
#         Args:
#             transform_list (list): The list of transformations to modify.
#             input_channels (int): The number of input channels in the dataset.
#         """
#         if input_channels == 1:
#             transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
#         else:
#             transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#
#     @staticmethod
#     def verify_labels(dataset):
#         """
#         Verifies and corrects labels in the dataset.
#
#         Args:
#             dataset (torch.utils.data.Dataset): The dataset to verify.
#
#         Returns:
#             torch.utils.data.Dataset: The dataset with corrected labels.
#         """
#         label_counts = Counter()
#         for _, label in dataset:
#             label_counts[label] += 1
#         majority_label = label_counts.most_common(1)[0][0]
#         corrected_labels = [majority_label if label != majority_label else label for _, label in dataset]
#         dataset.labels = corrected_labels
#         return dataset
#
#     def summarize_dataset(self, train_dataset, val_dataset=None, test_dataset=None):
#         """
#         Logs summary statistics of the dataset, including sample counts and basic statistics.
#
#         Args:
#             train_dataset (torch.utils.data.Dataset): The training dataset.
#             val_dataset (torch.utils.data.Dataset, optional): The validation dataset. Defaults to None.
#             test_dataset (torch.utils.data.Dataset, optional): The test dataset. Defaults to None.
#         """
#         logging.info(f"Summarizing dataset: {self.dataset_name}")
#         logging.info(f"Number of training samples: {len(train_dataset)}")
#         if val_dataset is not None:
#             logging.info(f"Number of validation samples: {len(val_dataset)}")
#         if test_dataset is not None:
#             logging.info(f"Number of test samples: {len(test_dataset)}")
#         # self.print_class_counts(train_dataset)
#         self.calculate_basic_statistics(train_dataset)
#         self.visualize_samples(self.model_type, train_dataset)
#
#     @staticmethod
#     def print_class_counts(dataset):
#         """
#         Calculates and logs the counts of each class in the dataset.
#
#         Args:
#             dataset (torch.utils.data.Dataset): The dataset to calculate class counts for.
#
#         Returns:
#             tuple: A tuple containing:
#                 - classes (list or None): The list of class names, or None if the classes attribute is not present.
#                 - class_counts (list or None): The list of counts for each class, or None if the classes attribute
#                 is not present.
#         """
#         logging.info("Calculating class counts")
#         # Handle if the dataset is a Subset
#         if isinstance(dataset, torch.utils.data.Subset):
#             original_dataset = dataset.dataset
#         else:
#             original_dataset = dataset
#
#         # Check if the 'classes' attribute exists
#         if hasattr(original_dataset, 'classes'):
#             classes = original_dataset.classes
#         else:
#             logging.warning("The dataset does not have a 'classes' attribute.")
#             # Handle the case where 'classes' does not exist, e.g., by inferring classes or skipping
#             return None, None
#
#         indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else range(len(original_dataset))
#         # Initialize the class counts
#         class_counts = [0] * len(classes)
#         # Count the occurrences of each class in the subset
#         for idx in indices:
#             _, label = original_dataset[idx]
#             class_counts[label] += 1
#
#         logging.info(f"Classes: {classes}")
#         logging.info(f"Class counts: {class_counts}")
#
#         return classes, class_counts
#
#     def extract_classes(self, dataset):
#         """
#         Extracts and returns the classes from the dataset.
#
#         Args:
#             dataset (torch.utils.data.Dataset): The dataset to extract classes from.
#
#         Returns:
#             list: The list of class names.
#         """
#         classes, _ = self.print_class_counts(dataset)
#         if classes is None:
#             raise ValueError("Could not determine the number of classes.")
#         return classes
#
#     @staticmethod
#     def calculate_basic_statistics(dataset):
#         """
#         Calculates and logs basic statistics such as mean, median, and standard deviation.
#
#         Args:
#             dataset (torch.utils.data.Dataset): The dataset to calculate statistics for.
#         """
#         data_key = []
#         for data, _ in dataset:
#             data_key.append(data.numpy())
#         data_array = np.array(data_key)
#         mean = np.mean(data_array, axis=0)
#         median = np.median(data_array, axis=0)
#         std_dev = np.std(data_array, axis=0)
#         global_mean = np.mean(mean)
#         global_median = np.mean(median)
#         global_std_dev = np.mean(std_dev)
#         logging.info(
#             f"Global Mean: {global_mean}, Global Median: {global_median}, Global Standard Deviation: {global_std_dev}")
#
#     def visualize_samples(self, model_name, dataset, num_samples=5):
#         """
#         Visualizes and saves a set of sample images from the dataset.
#
#         Args:
#             model_name (str): The name of the model.
#             dataset (torch.utils.data.Dataset): The dataset to visualize samples from.
#             num_samples (int, optional): The number of samples to visualize. Defaults to 5.
#         """
#         output_dir = os.path.join('out', self.task_name, self.dataset_name, 'pre_visualization')
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Accumulate all samples and labels
#         sample_images = []
#         sample_labels = []
#         for i in range(num_samples):
#             img, label = dataset[i]
#             if isinstance(img, torch.Tensor):
#                 img = img.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
#             if img.shape[2] == 1:  # Grayscale image
#                 img = img.squeeze()
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
#             sample_images.append(img)
#             sample_labels.append(label)
#
#         # Create a single figure for visualization
#         fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
#         for i in range(num_samples):
#             axs[i].imshow(sample_images[i])
#             axs[i].set_title(f'Model: {model_name} | Label: {sample_labels[i]}')
#             axs[i].axis('off')
#
#         # Save the complete visualization
#         output_path = os.path.join(output_dir, f'sample_visualization_model_{model_name}.png')
#         plt.savefig(output_path)
#         logging.info(f'Complete sample visualization saved to {output_path}')
#         plt.close(fig)

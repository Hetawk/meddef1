import os
import logging
import shutil
from pathlib import Path
from typing import Tuple, Dict
import torch
from torchvision import transforms  # Add this import
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
from .preprocessing import build_transforms, check_for_corrupted_images, get_default_transforms
from PIL import Image
import numpy as np


class DatasetHandler:
    def __init__(self, dataset_name: str, config_path="./loader/config.yaml"):
        self.dataset_name = dataset_name
        try:
            self.config = self._load_config(config_path)
            self.dataset_config = self._get_dataset_config()
            self.structure_type = self._get_structure_type()
            self.transforms = self._get_transforms()
            self.final_transforms = get_default_transforms(
                self.config.get('common_settings', {}))
        except Exception as e:
            logging.error(f"Failed to initialize DatasetHandler: {str(e)}")
            raise

    def _load_config(self, config_path):
        """Load and validate config"""
        if not os.path.exists(config_path):
            alt_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            if not os.path.exists(alt_path):
                raise FileNotFoundError(
                    f"Config file not found at {config_path} or {alt_path}")
            config_path = alt_path

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'data' not in config:
            raise KeyError("Config file must contain 'data' section")
        return config['data']

    def _get_dataset_config(self):
        """Get dataset specific configuration"""
        for dataset in self.config.get('data_key', []):
            if dataset['name'] == self.dataset_name:
                return dataset
        raise ValueError(f"Dataset {self.dataset_name} not found in config")

    def _get_structure_type(self):
        """Determine dataset structure type"""
        for structure_type, datasets in self.config.get('dataset_structure', {}).items():
            if self.dataset_name in datasets:
                return structure_type
        return "standard"

    def _get_transforms(self):
        """Get dataset specific transforms"""
        transforms = {}
        for mode in ['train', 'val', 'test']:
            transforms[mode] = build_transforms(self.dataset_name, mode)
        return transforms

    def process_and_load(self, output_dir: str,
                         train_batch_size: int,
                         val_batch_size: int,
                         test_batch_size: int,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Process dataset and return dataloaders"""
        output_path = Path(output_dir) / self.dataset_name

        # Process dataset structure
        self._process_structure(output_path)

        # Load processed data
        train_dataset = ImageFolder(
            output_path / 'train', self.transforms['train'])
        val_dataset = ImageFolder(output_path / 'val', self.transforms['val'])
        test_dataset = ImageFolder(
            output_path / 'test', self.transforms['test'])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def _process_structure(self, output_dir: Path):
        """Process dataset according to its structure type"""
        if self.structure_type == "class_based":
            self._process_class_based(output_dir)
        elif self.structure_type == "train_test":
            self._process_train_test(output_dir)
        elif self.structure_type == "train_valid_test":
            self._process_train_valid_test(output_dir)
        else:
            self._process_standard(output_dir)

    def _get_image_properties(self, image_path: Path, is_processed: bool = False) -> Dict:
        """Get image properties showing both original and processed states"""
        with Image.open(image_path) as img:
            original_size = img.size
            original_channels = len(img.getbands())
            original_bands = img.getbands()

            if is_processed:
                return {
                    'size': original_size,
                    'channels': original_channels,
                    'bands': original_bands
                }

            # For original image, apply transforms to get processed properties
            transformed_img = self.final_transforms['train'](img)
            processed_shape = tuple(transformed_img.shape)

            return {
                'original': {
                    'size': original_size,
                    'channels': original_channels,
                    'bands': original_bands
                },
                'processed': {
                    'shape': processed_shape,
                    'channels': processed_shape[0],
                }
            }

    def _log_dataset_info(self, output_dir: Path, split_info: Dict):
        """Log dataset information to a file with both original and processed image properties"""
        info_file = output_dir / 'dataset_info.txt'
        with open(info_file, 'w') as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write("=" * 50 + "\n\n")

            # Get original image properties from source directory
            source_path = Path(self.config['data_dir']) / self.dataset_name
            first_image = None

            try:
                if self.structure_type == "class_based":
                    # iterate over class directories until an image is found
                    for class_dir in source_path.iterdir():
                        if class_dir.is_dir():
                            images = list(class_dir.glob('*.*'))
                            if images:
                                first_image = images[0]
                                break
                else:
                    train_dir = source_path / "train"
                    if train_dir.exists():
                        for class_dir in train_dir.iterdir():
                            if class_dir.is_dir():
                                images = list(class_dir.glob('*.*'))
                                if images:
                                    first_image = images[0]
                                    break
            except Exception as e:
                logging.error(f"Error retrieving first image: {str(e)}")

            if first_image:
                original_props = self._get_image_properties(first_image)
                f.write("Original Image Properties:\n")
                f.write(f"  - Size: {original_props['original']['size']}\n")
                f.write(
                    f"  - Channels: {original_props['original']['channels']} ({', '.join(original_props['original']['bands'])})\n\n")
                f.write("Target Processing Properties:\n")
                f.write(f"  - Shape: {original_props['processed']['shape']}\n")
                f.write(
                    f"  - Channels: {original_props['processed']['channels']}\n\n")
            else:
                f.write("No original image found to display properties.\n\n")

            # Write split information
            f.write("Split Information:\n")
            f.write("-" * 20 + "\n")
            total_images = 0

            for split, info in split_info.items():
                f.write(f"\n{split.upper()}:\n")
                f.write(f"Total images: {info.get('total', 0)}\n")
                f.write("Classes:\n")
                split_path = output_dir / split
                try:
                    first_class_path = next(split_path.iterdir())
                    first_proc_images = list(first_class_path.glob('*.*'))
                    if first_proc_images:
                        first_proc_image = first_proc_images[0]
                        proc_props = self._get_image_properties(
                            first_proc_image, True)
                        f.write("Processed Image Properties:\n")
                        f.write(f"  - Size: {proc_props.get('size', 'N/A')}\n")
                        # For backward compatibility, check for 'bands'
                        channels = proc_props.get('channels', 'N/A')
                        bands = proc_props.get('bands', [])
                        if bands:
                            f.write(
                                f"  - Channels: {channels} ({', '.join(bands)})\n")
                        else:
                            f.write(f"  - Channels: {channels}\n")
                    else:
                        f.write("No processed image found in this split.\n")
                except StopIteration:
                    f.write("No class directories found in this split.\n")
                f.write("Class Distribution:\n")
                for class_name, count in info.get('classes', {}).items():
                    f.write(f"  - {class_name}: {count} images\n")
                total_images += info.get('total', 0)

            f.write(f"\nTotal Dataset Images: {total_images}\n")

            # Add preprocessing information
            f.write("\nPreprocessing Information:\n")
            f.write("-" * 20 + "\n")
            preproc_config = self.dataset_config.get('preprocessing', {})
            aug_config = self.dataset_config.get('augmentation', {})

            if preproc_config:
                f.write("Applied Preprocessing:\n")
                for key, value in preproc_config.items():
                    f.write(f"  - {key}: {value}\n")

            if aug_config:
                f.write("Augmentation Settings:\n")
                for key, value in aug_config.items():
                    f.write(f"  - {key}: {value}\n")

            f.write("=" * 50 + "\n")

    def _copy_and_transform_files(self, files, dest_dir: Path, desc: str, mode='train'):
        """Copy files with standardized transformations"""
        class_counts = {}
        dest_dir.mkdir(parents=True, exist_ok=True)

        transform = self.final_transforms[mode]

        for f in tqdm(files, desc=desc, unit='file'):
            class_name = f.parent.name
            dest_class_dir = dest_dir / class_name
            dest_class_dir.mkdir(exist_ok=True)

            # Load and transform image
            with Image.open(f) as img:
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB
                    img = img.convert('RGB')
                # If conversion is requested, convert non-RGB images to RGB (3 channels)
                if self.dataset_config.get("conversion", "") == "convert_to_3_channel" and img.mode != "RGB":
                    img = img.convert("RGB")
                # Apply transforms
                transformed_img = transform(img)
                # Convert tensor back to PIL for saving
                transformed_img = transforms.ToPILImage()(transformed_img)
                # Save with original name
                transformed_img.save(dest_class_dir / f.name)

            # Update class counts
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return class_counts

    def _process_class_based(self, output_dir: Path):
        """Handle datasets like tbcr with class-based structure"""
        dataset_path = Path(self.config['data_dir']) / self.dataset_name
        split_ratios = self.dataset_config['structure'].get(
            'split_ratios', [0.7, 0.15, 0.15])
        split_info = {'train': {}, 'val': {}, 'test': {}}

        logging.info(
            f"Processing {self.dataset_name} dataset (class-based structure)")

        # Get classes from structure configuration
        classes = self.dataset_config.get('structure', {}).get('classes', [])
        if not classes:
            # Fallback to auto-detecting classes from directory
            classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
            logging.info(f"Auto-detected classes: {classes}")

        for class_name in tqdm(classes, desc="Processing classes"):
            class_path = dataset_path / class_name
            if not class_path.exists():
                logging.error(f"Class directory not found: {class_path}")
                continue

            # Get all image files (excluding metadata files)
            files = [f for f in class_path.glob('*.*')
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                     and not f.name.endswith('.xlsx')]

            if not files:
                logging.warning(f"No image files found in {class_path}")
                continue

            if len(files) < 2:
                logging.warning(
                    f"Not enough images in {class_name} to split into train/val; assigning all images to training.")
                train_files = files
                val_files = []  # No validation split
            else:
                try:
                    train_files, test_files = train_test_split(
                        files, test_size=split_ratios[2], random_state=42)
                    # If there is only one image in train_files, skip splitting further
                    if len(train_files) < 2:
                        logging.warning(
                            f"Not enough training images in {class_name} after test split; assigning all to training.")
                        val_files = []
                    else:
                        train_files, val_files = train_test_split(
                            train_files,
                            test_size=split_ratios[1] /
                            (split_ratios[0]+split_ratios[1]),
                            random_state=42
                        )
                except Exception as e:
                    logging.error(
                        f"Error processing class {class_name}: {str(e)}")
                    raise

            # Process each split: if validation split is empty, skip its processing
            for split_name, split_files in [
                ('train', train_files),
                ('val', val_files),
                ('test', test_files) if len(files) >= 2 else ('test', [])
            ]:
                if not split_files:
                    logging.info(
                        f"No files for {split_name} in class {class_name}; skipping.")
                    continue
                split_dir = output_dir / split_name
                class_counts = self._copy_and_transform_files(
                    split_files,
                    split_dir,
                    f"Processing {split_name} - {class_name}",
                    mode=split_name
                )
                split_info[split_name].setdefault(
                    'classes', {}).update(class_counts)
                split_info[split_name]['total'] = sum(
                    split_info[split_name]['classes'].values())

        # Log dataset information
        self._log_dataset_info(output_dir, split_info)
        logging.info(f"Completed processing {self.dataset_name}")

    def _process_train_test(self, output_dir: Path):
        """Handle datasets with only train/test splits"""
        dataset_path = Path(self.config['data_dir']) / self.dataset_name
        structure = self.dataset_config['structure']
        split_info = {'train': {}, 'val': {}, 'test': {}}

        logging.info(
            f"Processing {self.dataset_name} dataset (train-test structure)")

        # Process train directory
        train_dir = dataset_path / structure['train']
        total_classes = len([d for d in train_dir.iterdir() if d.is_dir()])

        for class_dir in tqdm(train_dir.iterdir(), desc="Processing classes", total=total_classes):
            if class_dir.is_dir():
                files = list(class_dir.glob('*.*'))
                train_files, val_files = train_test_split(
                    files, train_size=0.85, random_state=42)

                # Process train split
                train_counts = self._copy_and_transform_files(
                    train_files,
                    output_dir / 'train',
                    f"Processing train - {class_dir.name}",
                    mode='train'
                )
                split_info['train'].setdefault(
                    'classes', {}).update(train_counts)
                split_info['train']['total'] = sum(
                    split_info['train']['classes'].values())

                # Process validation split
                val_counts = self._copy_and_transform_files(
                    val_files,
                    output_dir / 'val',
                    f"Processing validation - {class_dir.name}",
                    mode='val'
                )
                split_info['val'].setdefault('classes', {}).update(val_counts)
                split_info['val']['total'] = sum(
                    split_info['val']['classes'].values())

        # Process test directory
        test_dir = dataset_path / structure['test']
        if test_dir.exists():
            for class_dir in tqdm(test_dir.iterdir(), desc="Processing test set"):
                if class_dir.is_dir():
                    test_counts = self._copy_and_transform_files(
                        list(class_dir.glob('*.*')),
                        output_dir / 'test',
                        f"Processing test - {class_dir.name}",
                        mode='test'
                    )
                    split_info['test'].setdefault(
                        'classes', {}).update(test_counts)
                    split_info['test']['total'] = sum(
                        split_info['test']['classes'].values())

        # Log dataset information
        self._log_dataset_info(output_dir, split_info)
        logging.info(f"Completed processing {self.dataset_name}")

    def _process_train_valid_test(self, output_dir: Path):
        """Handle datasets with train/valid/test structure"""
        dataset_path = Path(self.config['data_dir']) / self.dataset_name
        split_info = {'train': {}, 'val': {}, 'test': {}}

        logging.info(
            f"Processing {self.dataset_name} dataset (train-valid-test structure)")

        # Map source directories to destination directories
        dir_mapping = {
            'train': 'train',
            'valid': 'val',
            'test': 'test'
        }

        for src_name, dst_name in dir_mapping.items():
            src_dir = dataset_path / src_name
            if src_dir.exists():
                logging.info(f"Processing {src_name} directory...")
                total_classes = len(
                    [d for d in src_dir.iterdir() if d.is_dir()])

                for class_dir in tqdm(src_dir.iterdir(), desc=f"Processing {src_name}", total=total_classes):
                    if class_dir.is_dir():
                        counts = self._copy_and_transform_files(
                            list(class_dir.glob('*.*')),
                            output_dir / dst_name,
                            f"Processing {src_name} - {class_dir.name}",
                            mode=dst_name
                        )
                        split_info[dst_name].setdefault(
                            'classes', {}).update(counts)
                        split_info[dst_name]['total'] = sum(
                            split_info[dst_name]['classes'].values())

        # Log dataset information
        self._log_dataset_info(output_dir, split_info)
        logging.info(f"Completed processing {self.dataset_name}")

    def _process_standard(self, output_dir: Path):
        """Handle datasets with standard train/val/test structure"""
        dataset_path = Path(self.config['data_dir']) / self.dataset_name
        split_info = {'train': {}, 'val': {}, 'test': {}}

        logging.info(
            f"Processing {self.dataset_name} dataset (standard structure)")

        for split in ['train', 'val', 'test']:
            src_dir = dataset_path / split
            if src_dir.exists():
                logging.info(f"Processing {split} directory...")
                total_classes = len(
                    [d for d in src_dir.iterdir() if d.is_dir()])

                for class_dir in tqdm(src_dir.iterdir(), desc=f"Processing {split}", total=total_classes):
                    if class_dir.is_dir():
                        counts = self._copy_and_transform_files(
                            list(class_dir.glob('*.*')),
                            output_dir / split,
                            f"Processing {split} - {class_dir.name}",
                            mode=split
                        )
                        split_info[split].setdefault(
                            'classes', {}).update(counts)
                        split_info[split]['total'] = sum(
                            split_info[split]['classes'].values())

        # Log dataset information
        self._log_dataset_info(output_dir, split_info)
        logging.info(f"Completed processing {self.dataset_name}")

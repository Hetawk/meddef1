import os
import logging
import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
from .preprocessing import build_transforms, check_for_corrupted_images, get_default_transforms
from PIL import Image
import numpy as np
import re


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
                         pin_memory: bool = True,
                         enforce_split_ratio: bool = True,
                         split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Process dataset with enforced train/val/test split and return dataloaders"""
        output_path = Path(output_dir) / self.dataset_name

        # Process dataset structure with enforced split ratios
        self._process_structure(
            output_path, enforce_split_ratio=enforce_split_ratio, split_ratio=split_ratio)

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

    def _process_structure(self, output_dir: Path,
                           enforce_split_ratio: bool = True,
                           split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Process dataset according to its structure type with enforced split ratios"""
        if enforce_split_ratio:
            logging.info(
                f"Enforcing {split_ratio[0]*100:.0f}/{split_ratio[1]*100:.0f}/{split_ratio[2]*100:.0f} train/val/test split ratio")
            self._process_with_enforced_split(output_dir, split_ratio)
        else:
            # Use original processing methods based on structure type
            if self.structure_type == "class_based":
                self._process_class_based(output_dir)
            elif self.structure_type == "train_test":
                self._process_train_test(output_dir)
            elif self.structure_type == "train_valid_test":
                self._process_train_valid_test(output_dir)
            else:
                self._process_standard(output_dir)

    def _normalize_class_name(self, class_name: str) -> str:
        """
        Normalize class names by extracting the base class name.
        e.g., "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib" -> "adenocarcinoma"

        Rules:
        1. Use dataset-specific rules defined in config if available
        2. Otherwise, use general rules:
           - Split on first underscore or dot
           - Use predefined mappings if available
        """
        # Check if there's a specific normalization rule for this dataset
        if 'class_normalization' in self.dataset_config:
            norm_rules = self.dataset_config['class_normalization']

            # Direct mapping if available
            if 'mapping' in norm_rules and class_name in norm_rules['mapping']:
                return norm_rules['mapping'][class_name]

            # Pattern-based replacement
            if 'patterns' in norm_rules:
                for pattern, replacement in norm_rules['patterns'].items():
                    if re.search(pattern, class_name):
                        return re.sub(pattern, replacement, class_name)

            # Delimiter-based splitting
            if 'delimiter' in norm_rules:
                delimiter = norm_rules['delimiter']
                return class_name.split(delimiter)[0]

        # General fallback rules

        # For CCTS dataset format like "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"
        if '_' in class_name:
            return class_name.split('_')[0]

        # For names with dots like "large.cell.carcinoma"
        # Keep the full name with dots as it may be a compound class name

        return class_name

    def _copy_and_transform_files(self, files, dest_dir: Path, desc: str, mode='train'):
        """Copy files with standardized transformations"""
        class_counts = {}
        dest_dir.mkdir(parents=True, exist_ok=True)

        transform = self.final_transforms[mode]

        for f in tqdm(files, desc=desc, unit='file'):
            # Get the original class name (parent directory name)
            original_class_name = f.parent.name

            # Normalize the class name for consistency
            normalized_class_name = self._normalize_class_name(
                original_class_name)

            # Use the normalized class name for the destination directory
            dest_class_dir = dest_dir / normalized_class_name
            dest_class_dir.mkdir(exist_ok=True)

            # Load and transform image
            with Image.open(f) as img:
                # Always convert to RGB regardless of mode
                # This ensures consistent channels across train/val/test
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Apply transforms
                transformed_img = transform(img)
                # Convert tensor back to PIL for saving
                transformed_img = transforms.ToPILImage()(transformed_img)
                # Save with original name
                transformed_img.save(dest_class_dir / f.name)

            # Update class counts with normalized class name
            class_counts[normalized_class_name] = class_counts.get(
                normalized_class_name, 0) + 1

        return class_counts

    def _process_with_enforced_split(self, output_dir: Path,
                                     split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Process any dataset structure with enforced train/val/test split ratios"""
        dataset_path = Path(self.config['data_dir']) / self.dataset_name
        split_info = {'train': {}, 'val': {}, 'test': {}}

        # First pass: collect all images by class
        all_images_by_class = {}

        logging.info(
            f"Processing {self.dataset_name} with enforced {split_ratio[0]*100:.0f}/{split_ratio[1]*100:.0f}/{split_ratio[2]*100:.0f} split")

        # Handle different dataset structures for collection
        if self.structure_type == "class_based":
            # Get classes from structure configuration or auto-detect
            classes = self.dataset_config.get(
                'structure', {}).get('classes', [])
            if not classes:
                classes = [d.name for d in dataset_path.iterdir()
                           if d.is_dir()]
                logging.info(f"Auto-detected classes: {classes}")

            for class_name in classes:
                class_path = dataset_path / class_name
                if not class_path.exists():
                    logging.warning(f"Class directory not found: {class_path}")
                    continue

                images = [f for f in class_path.glob('*.*')
                          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                          and not f.name.endswith('.xlsx')]

                all_images_by_class[class_name] = images

        elif self.structure_type in ["train_test", "train_valid_test", "standard"]:
            # Collect images from all splits (train, val/valid, test)
            dirs_to_check = []

            if self.structure_type == "train_test":
                structure = self.dataset_config.get('structure', {})
                dirs_to_check = [dataset_path / structure.get('train', 'train'),
                                 dataset_path / structure.get('test', 'test')]
            elif self.structure_type == "train_valid_test":
                dirs_to_check = [dataset_path / 'train',
                                 dataset_path / 'valid',
                                 dataset_path / 'test']
            else:  # standard
                dirs_to_check = [dataset_path / 'train',
                                 dataset_path / 'val',
                                 dataset_path / 'test']

            # Find all classes across all splits, using normalized class names
            all_classes = set()
            normalized_class_mapping = {}  # Maps original dir names to normalized class names

            for split_dir in dirs_to_check:
                if not split_dir.exists():
                    continue
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        normalized_class_name = self._normalize_class_name(
                            class_dir.name)
                        all_classes.add(normalized_class_name)
                        normalized_class_mapping[class_dir.name] = normalized_class_name

            # Log class normalization info
            if len(normalized_class_mapping) > len(all_classes):
                logging.info(
                    f"Normalized {len(normalized_class_mapping)} class directories to {len(all_classes)} unique classes")
                for orig, norm in normalized_class_mapping.items():
                    if orig != norm:
                        logging.info(f"  Mapped '{orig}' → '{norm}'")

            # Collect images from each class across all splits
            for normalized_class_name in all_classes:
                all_images_by_class[normalized_class_name] = []

            # Now collect images using the mapping
            for split_dir in dirs_to_check:
                if not split_dir.exists():
                    continue

                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        normalized_class_name = normalized_class_mapping.get(
                            class_dir.name, class_dir.name)

                        images = [f for f in class_dir.glob('*.*')
                                  if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                                  and not f.name.endswith('.xlsx')]

                        all_images_by_class[normalized_class_name].extend(
                            images)

        # Second pass: create balanced splits for each class
        for class_name, images in all_images_by_class.items():
            if not images:
                logging.warning(f"No images found for class: {class_name}")
                continue

            # Ensure reproducible splits with fixed random state
            train_images, temp_images = train_test_split(
                images,
                train_size=split_ratio[0],
                random_state=42
            )

            # Further split the remaining data into val and test
            val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
            val_images, test_images = train_test_split(
                temp_images,
                train_size=val_ratio,
                random_state=42
            )

            # Process and save each split
            splits = [
                ('train', train_images),
                ('val', val_images),
                ('test', test_images)
            ]

            for split_name, split_images in splits:
                if not split_images:
                    logging.warning(
                        f"No images for {split_name} split in class {class_name}")
                    continue

                split_dir = output_dir / split_name
                class_counts = self._copy_and_transform_files(
                    split_images,
                    split_dir,
                    f"Processing {split_name} - {class_name}",
                    mode=split_name
                )

                split_info[split_name].setdefault(
                    'classes', {}).update(class_counts)
                split_info[split_name]['total'] = sum(
                    split_info[split_name]['classes'].values()
                )

        # Log dataset information
        self._log_dataset_info(output_dir, split_info)
        logging.info(
            f"Completed processing {self.dataset_name} with enforced split ratio")

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

    # Keep the original methods for backwards compatibility
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

        # Track class name normalization for logging
        normalized_class_mapping = {}

        for src_name, dst_name in dir_mapping.items():
            src_dir = dataset_path / src_name
            if src_dir.exists():
                logging.info(f"Processing {src_name} directory...")
                total_classes = len(
                    [d for d in src_dir.iterdir() if d.is_dir()])

                for class_dir in tqdm(src_dir.iterdir(), desc=f"Processing {src_name}", total=total_classes):
                    if class_dir.is_dir():
                        # Track normalization for logging
                        normalized_name = self._normalize_class_name(
                            class_dir.name)
                        if class_dir.name != normalized_name:
                            normalized_class_mapping[class_dir.name] = normalized_name

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

        # Log normalization results
        if normalized_class_mapping:
            logging.info(
                f"Normalized {len(normalized_class_mapping)} class directories:")
            for orig, norm in normalized_class_mapping.items():
                logging.info(f"  Mapped '{orig}' → '{norm}'")

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

import torch
from torch.utils.data import DataLoader, Dataset
import psutil
import os
import logging
from PIL import Image
from typing import Dict
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch.multiprocessing as mp
import ctypes


class FastDataLoader(DataLoader):
    """Optimized DataLoader with better memory management"""

    def __init__(self, dataset, **kwargs):
        # Force using the spawn method for Windows compatibility
        if kwargs.get('num_workers', 0) > 0:
            kwargs['multiprocessing_context'] = mp.get_context('spawn')
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = 2

        super().__init__(dataset, **kwargs)


class FastImageFolder(Dataset):
    """Memory efficient image folder with numpy mmap and shared memory"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        self._cached_images: Dict[str, np.ndarray] = {}
        self._max_cache_size = 1000
        self._load_dataset()

    def _load_dataset(self):
        # Pre-load class info and file paths
        logging.info(f"Loading dataset from {self.root}")
        for class_name in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.classes)
                self.classes.append(class_name)

            # Use numpy memmap for large files
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_path, img_name),
                                         self.class_to_idx[class_name]))
                    self.targets.append(self.class_to_idx[class_name])

    def __getitem__(self, index):
        path, target = self.samples[index]

        try:
            # Try to get from cache first
            if path in self._cached_images:
                img = self._cached_images[path]
            else:
                # Read image with PIL for speed
                img = Image.open(path).convert('RGB')

                # Cache if not too many images cached
                if len(self._cached_images) < self._max_cache_size:
                    self._cached_images[path] = img.copy()

            if self.transform is not None:
                img = self.transform(img)

            return img, target
        except Exception as e:
            logging.error(f"Error loading image {path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224), 'black')
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.samples)


def create_efficient_loaders(train_dataset, val_dataset=None, test_dataset=None,
                             batch_size=32, num_workers=4, pin_memory=True, device=None,
                             use_amp=False, optimize_memory=False, gradient_checkpointing=False):
    """Create optimized data loaders"""

    # Optimize CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if use_amp:
            # Enable TF32 for better performance with AMP
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Memory optimizations
        if optimize_memory:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Optimize number of workers
    cpu_count = psutil.cpu_count(logical=False) or 2
    optimal_workers = min(num_workers, cpu_count - 1, 6)  # Cap at 6 workers

    # Common loader settings
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': optimal_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': 2,
        'drop_last': True  # Slightly faster training
    }

    # Enable additional optimizations if requested
    if optimize_memory:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
        torch.backends.cudnn.benchmark = True

    loaders = {}

    if train_dataset is not None:
        loaders['train'] = FastDataLoader(
            dataset=train_dataset, shuffle=True, **loader_kwargs)

    # Use fewer workers for validation/test
    loader_kwargs['num_workers'] = max(1, optimal_workers // 2)
    loader_kwargs['drop_last'] = False

    if val_dataset is not None:
        loaders['val'] = FastDataLoader(
            dataset=val_dataset, shuffle=False, **loader_kwargs)

    if test_dataset is not None:
        loaders['test'] = FastDataLoader(
            dataset=test_dataset, shuffle=False, **loader_kwargs)

    return loaders

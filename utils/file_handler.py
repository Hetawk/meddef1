import os
import logging
from datetime import datetime
import torch
import pandas as pd


class FileHandler:
    """Handles all file operations with consistent path structure"""

    def __init__(self, task_name, dataset_name, model_name):
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.base_path = self._create_base_path()

    def _create_base_path(self):
        """Creates and returns base path for all saved files"""
        return os.path.join('out', self.task_name, self.dataset_name, self.model_name)

    def get_save_path(self, filename, subfolder=None, include_timestamp=True, **kwargs):
        """Generate full path with consistent naming pattern"""
        name, ext = os.path.splitext(filename)

        # Add metadata to filename if provided
        metadata = []
        for key, value in kwargs.items():
            metadata.append(f"{key}{value}")

        if metadata:
            name = f"{name}_{'_'.join(metadata)}"

        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{timestamp}"

        full_name = f"{name}{ext}"

        if subfolder:
            path = os.path.join(self.base_path, subfolder, full_name)
        else:
            path = os.path.join(self.base_path, full_name)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def save_model(self, model, filename="model.pth", **kwargs):
        """Save model with consistent path structure"""
        path = self.get_save_path(filename, subfolder="save_model", **kwargs)
        torch.save(model.state_dict(), path)
        logging.info(f'Model saved to {path}')
        return path

    def save_history(self, history, filename="training_history.csv"):
        """Save training history with consistent path structure"""
        path = self.get_save_path(filename, include_timestamp=False)

        df = pd.DataFrame(history)
        # Convert list columns to string representation
        for col in ['true_labels', 'predictions']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ','.join(map(str, x)))

        if not os.path.isfile(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode='a', header=False, index=False)

        logging.info(f'Training history saved to {path}')
        return path

    def load_model(self, model, filename="model.pth", device='cuda'):
        """Load model with consistent path structure"""
        path = os.path.join(self.base_path, "save_model", filename)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        logging.info(f"Loaded model from {path}")
        return model

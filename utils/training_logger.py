# training_logger.py

import os
import json
import time
import torch
import pandas as pd
from datetime import datetime
import logging


class TrainingLogger:
    def __init__(self, log_dir='out/'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logged = False
        self.logger = logging.getLogger(__name__)

    def _get_base_path(self, task_name, dataset_name, model_name):
        """Create consistent base path for all files"""
        return os.path.join(self.log_dir, task_name, dataset_name, model_name)

    def save_model(self, model, task_name, dataset_name, model_name, epochs, lr, batch_size):
        """Save model with consistent naming"""
        base_path = self._get_base_path(task_name, dataset_name, model_name)
        save_dir = os.path.join(base_path, 'save_model')
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d")
        filename = f"best_{model_name}_{dataset_name}_epochs{epochs}_lr{lr}_batch{batch_size}_{timestamp}.pth"
        path = os.path.join(save_dir, filename)

        torch.save(model.state_dict(), path)
        logging.info(f'Model saved to {path}')

    def save_history(self, history, task_name, dataset_name, model_name):
        """Save training history"""
        base_path = self._get_base_path(task_name, dataset_name, model_name)
        path = os.path.join(base_path, 'training_history.csv')

        df = pd.DataFrame(history)
        for col in ['true_labels', 'predictions']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ','.join(map(str, x)))

        mode = 'a' if os.path.exists(path) else 'w'
        header = not os.path.exists(path)
        df.to_csv(path, mode=mode, header=header, index=False)
        logging.info(f'Training history saved to {path}')

    def log_training_info(self, task_name, model_name, dataset_name, hyperparams, metrics,
                          start_time, end_time, test_metrics, initial_params, final_params):
        """Log training information to text file"""
        if not self.logged:
            log_dir = os.path.join(
                self.log_dir, task_name, dataset_name, model_name)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "summary_training_log.txt")

            with open(log_file, 'a') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Start Time: {start_time}\n")
                if end_time:
                    duration = end_time - start_time
                    f.write(f"End Time: {end_time}\n")
                    f.write(f"Duration: {duration} seconds\n")
                f.write("\nHyperparameters:\n")
                for key, value in hyperparams.items():
                    f.write(f"{key}: {value}\n")
                if metrics:
                    f.write("\nMetrics:\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                if test_metrics:
                    f.write("\nTest Metrics:\n")
                    for key, value in test_metrics.items():
                        f.write(f"{key}: {value}\n")
                f.write(f"\nInitial Parameters: {initial_params:.2f}M\n")
                if final_params:
                    f.write(f"Final Parameters: {final_params:.2f}M\n")
                f.write("\n" + "-"*50 + "\n")

            self.logged = True

    def log_metrics(self, task_name, model_name, dataset_name, metrics):
        """Log metrics to text file"""
        log_dir = os.path.join(self.log_dir, task_name,
                               dataset_name, model_name)
        os.makedirs(log_dir, exist_ok=True)

        # Save to TXT
        log_file = os.path.join(log_dir, "summary_training_log.txt")
        with open(log_file, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("\nMetrics:\n")
            json.dump(metrics, f, indent=4)
            f.write("\n" + "-"*50 + "\n")

    

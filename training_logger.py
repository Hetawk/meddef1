# training_logger.py

import os
import json
import time

class TrainingLogger:
    def __init__(self, log_dir='out/'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_training_info(self, task_name, model_name, dataset_name, hyperparams, metrics, start_time, end_time, test_metrics):
        log_dir = os.path.join(self.log_dir, task_name, dataset_name, model_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "summary_training_log.txt")
        with open(log_file, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Start Time: {start_time}\n")
            f.write(f"End Time: {end_time}\n")
            f.write(f"Duration: {end_time - start_time} seconds\n")
            f.write("\nHyperparameters:\n")
            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")
            f.write("\nMetrics:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\nTest Metrics:\n")
            for key, value in test_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-"*50 + "\n")

    def log_metrics(self, task_name, model_name, dataset_name, metrics):
        log_dir = os.path.join(self.log_dir, task_name, dataset_name, model_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "summary_training_log.txt")
        with open(log_file, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("\nMetrics:\n")
            json.dump(metrics, f, indent=4)
            f.write("\n" + "-"*50 + "\n")
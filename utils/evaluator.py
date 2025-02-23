import pandas as pd
import os
import logging
import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from torch.nn import functional as F
from typing import List, Union, Optional
from utils.metrics import Metrics

class Evaluator:
    def __init__(self, model_name: str, results: List[dict], true_labels: Union[str, List[int]],
                 all_predictions: Union[str, List[Union[int, torch.Tensor]]], task_name: str,
                 all_probabilities: Optional[np.ndarray] = None):
        self.model_name = model_name
        self.results = results
        self.true_labels = true_labels
        self.all_predictions = all_predictions
        self.task_name = task_name
        self.all_probabilities = all_probabilities

    def evaluate(self, dataset_name: str):
        # Ensure true_labels and all_predictions are lists or arrays
        if isinstance(self.true_labels, str):
            self.true_labels = list(map(int, self.true_labels.split(',')))
        if isinstance(self.all_predictions, str):
            self.all_predictions = np.array(list(map(float, self.all_predictions.split(','))))
        elif isinstance(self.all_predictions, list):
            self.all_predictions = np.array(self.all_predictions)

        # Convert true_labels to NumPy array
        self.true_labels = np.array(self.true_labels)

        # Apply softmax to logits to get probabilities
        if self.all_predictions.ndim == 2 and self.all_predictions.shape[1] > 1:
            self.all_probabilities = F.softmax(torch.tensor(self.all_predictions), dim=1).numpy()
            predicted_classes = np.argmax(self.all_probabilities, axis=1)
        else:
            predicted_classes = self.all_predictions

        metrics = Metrics.calculate_metrics(self.true_labels, predicted_classes, self.all_probabilities)

        # Log and save metrics
        self.log_metrics(metrics)
        self.save_metrics(metrics, dataset_name)

        # Store results for each sample
        for i in range(len(self.true_labels)):
            self.results.append({
                'Model': self.model_name,
                'True Label': self.true_labels[i],
                'Predicted Label': predicted_classes[i]
            })

    def log_metrics(self, metrics: dict):
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")

    def save_metrics(self, metrics: dict, dataset_name: str):
        metrics_df = pd.DataFrame([{
            'Model': self.model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Precision Micro': metrics['precision_micro'],
            'Precision Weighted': metrics['precision_weighted'],
            'Recall': metrics['recall'],
            'Recall Micro': metrics['recall_micro'],
            'Recall Weighted': metrics['recall_weighted'],
            'F1 Score': metrics['f1'],
            'F1 Micro': metrics['f1_micro'],
            'F1 Weighted': metrics['f1_weighted'],
            'Specificity': metrics['specificity'],
            'Balanced Accuracy': metrics['balanced_accuracy'],
            'MCC': metrics['mcc'],
            'ROC AUC': metrics['roc_auc'],
            'Average Precision': metrics['average_precision'],
            'TP': metrics['tp'],
            'TN': metrics['tn'],
            'FP': metrics['fp'],
            'FN': metrics['fn']
        }])
        metrics_csv_path = os.path.join('out', self.task_name, dataset_name, self.model_name, f"all_evaluation_metrics.csv")
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

        if os.path.isfile(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0:
            existing_df = pd.read_csv(metrics_csv_path)
            metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)

        metrics_df.to_csv(metrics_csv_path, index=False)
        logging.info(f"Metrics saved to {metrics_csv_path}")

        return metrics

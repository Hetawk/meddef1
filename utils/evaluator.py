# evaluator.py

import pandas as pd
import os
import logging
from utils.metrics import Metrics
import pandas.errors

class Evaluator:
    def __init__(self, model_name, results, true_labels, all_predictions, task_name, all_probabilities=None):
        self.model_name = model_name
        self.results = results
        self.true_labels = true_labels
        self.all_predictions = all_predictions
        self.task_name = task_name
        self.all_probabilities = all_probabilities

    def evaluate(self, dataset_name):
        metrics = Metrics.calculate_metrics(self.true_labels, self.all_predictions, self.all_probabilities)

        # Log and save metrics to CSV
        self.log_metrics(metrics)
        self.save_metrics(metrics, dataset_name)


        # Add results to self.results
        for i in range(len(self.true_labels)):
            self.results.append({
                'Model': self.model_name,
                'True Label': self.true_labels[i],
                'Predicted Label': self.all_predictions[i]
            })

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")

    def save_metrics(self, metrics, dataset_name):
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
        metrics_csv_path = os.path.join('out', self.task_name, dataset_name,
                                        f"all_evaluation_metrics.csv")
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

        if os.path.isfile(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0:
            existing_df = pd.read_csv(metrics_csv_path)
            metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)

        metrics_df.to_csv(metrics_csv_path, index=False)
        logging.info(f"Metrics saved to {metrics_csv_path}")


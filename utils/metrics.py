# metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
import logging

class Metrics:
    @staticmethod
    def calculate_metrics(true_labels, all_predictions, all_probabilities=None):
        metrics = {}

        metrics['accuracy'] = accuracy_score(true_labels, all_predictions)
        metrics['precision'] = precision_score(true_labels, all_predictions, average='macro', zero_division=0)
        metrics['recall'] = recall_score(true_labels, all_predictions, average='macro', zero_division=0)
        metrics['f1'] = f1_score(true_labels, all_predictions, average='macro', zero_division=0)

        metrics['precision_micro'] = precision_score(true_labels, all_predictions, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(true_labels, all_predictions, average='weighted', zero_division=0)
        metrics['recall_micro'] = recall_score(true_labels, all_predictions, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(true_labels, all_predictions, average='weighted', zero_division=0)
        metrics['f1_micro'] = f1_score(true_labels, all_predictions, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(true_labels, all_predictions, average='weighted', zero_division=0)

        # Specificity calculation
        cm = confusion_matrix(true_labels, all_predictions)
        tn = cm.sum() - (cm.sum(axis=1) - cm.diagonal()).sum() - (
                cm.sum(axis=0) - cm.diagonal()).sum() + cm.diagonal().sum()
        metrics['specificity'] = tn / (tn + cm.sum(axis=1) - cm.diagonal()).sum()

        metrics['balanced_accuracy'] = balanced_accuracy_score(true_labels, all_predictions)
        metrics['mcc'] = matthews_corrcoef(true_labels, all_predictions)

        # Ensure the correct shape for ROC-AUC calculation
        if all_probabilities is not None and len(np.unique(true_labels)) == 2:
            metrics['roc_auc'] = roc_auc_score(true_labels, all_probabilities)
            metrics['average_precision'] = average_precision_score(true_labels, all_probabilities)
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None

        # Calculate TP, TN, FP, FN for multi-class confusion matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (fp + fn + tp)

        metrics['confusion_matrix'] = cm.tolist()  # Convert numpy array to list for JSON serialization
        metrics['tp'] = tp.tolist()
        metrics['tn'] = tn.tolist()
        metrics['fp'] = fp.tolist()
        metrics['fn'] = fn.tolist()

        return metrics

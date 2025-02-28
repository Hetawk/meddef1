import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, Optional


class Metrics:
    @staticmethod
    def specificity_score(true_labels, predicted_classes):
        # Specificity calculation
        cm = confusion_matrix(true_labels, predicted_classes)
        tn = cm.sum() - (cm.sum(axis=1) - cm.diagonal()).sum() - \
            (cm.sum(axis=0) - cm.diagonal()).sum() + cm.diagonal().sum()
        return tn / (tn + cm.sum(axis=1) - cm.diagonal()).sum()
    
    @staticmethod
    def expected_calibration_error(true_labels, predicted_classes, probabilities):
        # Calculate expected calibration error
        prob_max = np.max(probabilities, axis=1)
        correct = (predicted_classes == true_labels).astype(float)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (prob_max >= bins[i]) & (prob_max < bins[i + 1])
            if np.sum(bin_mask) > 0:
                avg_confidence = np.mean(prob_max[bin_mask])
                avg_accuracy = np.mean(correct[bin_mask])
                ece += np.abs(avg_confidence - avg_accuracy) * \
                    np.sum(bin_mask) / len(prob_max)
        return ece

    @classmethod
    def calculate_metrics(cls, y_true, y_pred, y_prob):
        metrics = {}
        # Handle different array formats
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Basic classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Additional classification metrics
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate specificity, balanced accuracy, and other metrics
        metrics['specificity'] = cls.specificity_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics with error handling
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                metrics['roc_auc'] = 0.0
            else:
                raise
                
        try:
            metrics['average_precision'] = average_precision_score(y_true, y_prob, average='macro')
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                metrics['average_precision'] = 0.0
            else:
                raise
        
        # Calculate log loss if applicable
        try:
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except ValueError:
            # If there's an issue with log loss, set a high value to indicate poor performance
            metrics['log_loss'] = 15.0  # Arbitrarily high value
        
        # Calculate Brier score and expected calibration error if applicable
        try:
            cls_names = np.unique(y_true)
            if len(cls_names) == 2:  # Binary classification
                metrics['brier_score'] = brier_score_loss(y_true, y_prob[:, 1])
            else:
                # For multiclass, we take the mean of per-class Brier scores
                brier_scores = []
                for i, cls_name in enumerate(cls_names):
                    binary_y_true = (y_true == cls_name).astype(int)
                    binary_y_prob = y_prob[:, i]
                    brier_scores.append(brier_score_loss(binary_y_true, binary_y_prob))
                metrics['brier_score'] = np.mean(brier_scores)
        except (ValueError, IndexError):
            metrics['brier_score'] = 1.0  # Worst possible Brier score
        
        # Expected Calibration Error
        try:
            metrics['ece'] = cls.expected_calibration_error(y_true, y_pred, y_prob)
        except:
            metrics['ece'] = 1.0  # Worst possible ECE
        
        return metrics

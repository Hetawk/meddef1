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

    @staticmethod
    def calculate_metrics(true_labels: np.ndarray,
                          all_predictions: np.ndarray,
                          all_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        metrics['accuracy'] = accuracy_score(true_labels, all_predictions)
        metrics['precision'] = precision_score(
            true_labels, all_predictions, average='macro', zero_division=0)
        metrics['recall'] = recall_score(
            true_labels, all_predictions, average='macro', zero_division=0)
        metrics['f1'] = f1_score(
            true_labels, all_predictions, average='macro', zero_division=0)

        metrics['precision_micro'] = precision_score(
            true_labels, all_predictions, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(
            true_labels, all_predictions, average='weighted', zero_division=0)
        metrics['recall_micro'] = recall_score(
            true_labels, all_predictions, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(
            true_labels, all_predictions, average='weighted', zero_division=0)
        metrics['f1_micro'] = f1_score(
            true_labels, all_predictions, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(
            true_labels, all_predictions, average='weighted', zero_division=0)

        metrics['specificity'] = Metrics.specificity_score(true_labels, all_predictions)
        metrics['balanced_accuracy'] = balanced_accuracy_score(true_labels, all_predictions)
        metrics['mcc'] = matthews_corrcoef(true_labels, all_predictions)
        metrics['cohen_kappa'] = cohen_kappa_score(true_labels, all_predictions)
        
        if all_probabilities is not None:
            # Handle ROC AUC differently for binary vs multiclass
            num_classes = all_probabilities.shape[1]
            if num_classes == 2:
                # For binary classification, use only the probability of the positive class (index 1)
                metrics['roc_auc'] = roc_auc_score(
                    true_labels, 
                    all_probabilities[:, 1]  # Use only probabilities for positive class
                )
                
                # For binary classification, also handle average precision differently
                metrics['average_precision'] = average_precision_score(
                    true_labels,
                    all_probabilities[:, 1]  # Use only probabilities for positive class
                )
            else:
                # For multiclass, binarize labels and use all probabilities
                true_labels_binarized = label_binarize(
                    true_labels, classes=np.arange(num_classes))
                metrics['roc_auc'] = roc_auc_score(
                    true_labels_binarized, all_probabilities, 
                    average='macro', multi_class='ovr'
                )
                
                # For multiclass, use binarized labels for average precision
                metrics['average_precision'] = average_precision_score(
                    true_labels_binarized, all_probabilities, average='macro'
                )
            
            metrics['log_loss'] = log_loss(true_labels, all_probabilities)
            
            # Brier Score Loss (only for binary classification)
            if all_probabilities.shape[1] == 2:
                try:
                    metrics['brier_score'] = brier_score_loss(
                        true_labels, all_probabilities[:, 1])
                except:
                    metrics['brier_score'] = None
            else:
                metrics['brier_score'] = None
                
            # Expected calibration error
            metrics['ece'] = Metrics.expected_calibration_error(
                true_labels, all_predictions, all_probabilities)
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
            metrics['log_loss'] = None
            metrics['brier_score'] = None
            metrics['ece'] = None

        # Calculate confusion matrix and related metrics
        cm = confusion_matrix(true_labels, all_predictions)
        metrics['confusion_matrix'] = cm.tolist()  # For JSON serialization
        
        # TP, TN, FP, FN for each class
        n_classes = len(np.unique(true_labels))
        tp = []
        tn = []
        fp = []
        fn = []
        
        for i in range(n_classes):
            tp_i = cm[i, i]
            fp_i = np.sum(cm[:, i]) - tp_i
            fn_i = np.sum(cm[i, :]) - tp_i
            tn_i = np.sum(cm) - tp_i - fp_i - fn_i
            
            tp.append(int(tp_i))
            tn.append(int(tn_i))
            fp.append(int(fp_i))
            fn.append(int(fn_i))
        
        metrics['tp'] = tp
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn

        return metrics

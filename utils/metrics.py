import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, Optional

class Metrics:
    @staticmethod
    def calculate_metrics(true_labels: np.ndarray,
                          all_predictions: np.ndarray,
                          all_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

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

        # Ensure the correct shape for ROC-AUC and average precision calculation
        if all_probabilities is not None:
            # Binarize the true labels for multi-class classification
            true_labels_binarized = label_binarize(true_labels, classes=np.unique(true_labels))
            metrics['roc_auc'] = roc_auc_score(true_labels_binarized, all_probabilities, average='macro', multi_class='ovr')
            metrics['average_precision'] = average_precision_score(true_labels_binarized, all_probabilities, average='macro')
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

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, Optional
import logging


class Metrics:
    @staticmethod
    def to_numpy(x: Any) -> np.ndarray:
        """Convert input to numpy array; supports torch tensors."""
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        return np.array(x)

    @staticmethod
    def calculate_metrics(true_labels: Any,
                          all_predictions: Any,
                          all_probabilities: Optional[Any] = None) -> Dict[str, Any]:
        """
        Calculate a suite of evaluation metrics.
        
        This version ensures that if the true labels are one-hot encoded,
        they are converted to a 1D array. It also handles both binary and
        multi-class cases robustly.
        """
        # Convert inputs to numpy arrays (handles torch tensors as well)
        true_labels = Metrics.to_numpy(true_labels)
        all_predictions = Metrics.to_numpy(all_predictions)
        
        # --- Convert true_labels to 1D if they are one-hot encoded ---
        if true_labels.ndim > 1:
            # Check if each row sums to 1 (one-hot check)
            if np.allclose(true_labels.sum(axis=1), 1, atol=1e-5):
                true_labels = np.argmax(true_labels, axis=1)
            else:
                true_labels = true_labels.flatten()

        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, all_predictions),
            'precision': precision_score(true_labels, all_predictions, average='macro', zero_division=0),
            'recall': recall_score(true_labels, all_predictions, average='macro', zero_division=0),
            'f1': f1_score(true_labels, all_predictions, average='macro', zero_division=0),
            'precision_micro': precision_score(true_labels, all_predictions, average='micro', zero_division=0),
            'precision_weighted': precision_score(true_labels, all_predictions, average='weighted', zero_division=0),
            'recall_micro': recall_score(true_labels, all_predictions, average='micro', zero_division=0),
            'recall_weighted': recall_score(true_labels, all_predictions, average='weighted', zero_division=0),
            'f1_micro': f1_score(true_labels, all_predictions, average='micro', zero_division=0),
            'f1_weighted': f1_score(true_labels, all_predictions, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(true_labels, all_predictions),
            'mcc': matthews_corrcoef(true_labels, all_predictions),
            'cohen_kappa': cohen_kappa_score(true_labels, all_predictions)
        }
        
        # Compute confusion matrix and derive TP, FP, FN, TN
        cm = confusion_matrix(true_labels, all_predictions)
        metrics['confusion_matrix'] = cm.tolist()
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        # Compute per-class specificity and then average them
        specificity = np.mean(tn / (tn + fp + 1e-10))
        metrics['specificity'] = specificity
        metrics['tp'] = tp.tolist()
        metrics['tn'] = tn.tolist()
        metrics['fp'] = fp.tolist()
        metrics['fn'] = fn.tolist()
        
        # Initialize advanced metrics with default values
        metrics['roc_auc'] = None
        metrics['average_precision'] = None
        metrics['log_loss'] = None
        metrics['brier_score'] = None
        metrics['ece'] = None
        
        if all_probabilities is not None:
            # Convert probabilities to numpy array (handle torch tensors as well)
            all_probabilities = Metrics.to_numpy(all_probabilities)
            
            try:
                # Determine the number of unique classes from true labels
                unique_classes = np.unique(true_labels)
                n_classes = len(unique_classes)
                
                # Calculate log loss (works for both binary and multi-class)
                try:
                    metrics['log_loss'] = log_loss(true_labels, all_probabilities)
                except Exception as e:
                    logging.warning(f"Log loss calculation error: {str(e)}")
                    metrics['log_loss'] = np.nan
                
                if n_classes == 2:
                    # --- Binary classification case ---
                    # For binary, we need the probability of the positive class (class 1)
                    if all_probabilities.ndim == 2 and all_probabilities.shape[1] == 2:
                        # Extract probability for positive class (class 1)
                        pos_probs = all_probabilities[:, 1]
                    else:
                        # If shape is not what we expect, flatten the array
                        pos_probs = all_probabilities.ravel()
                    
                    try:
                        # Compute binary metrics
                        metrics['roc_auc'] = roc_auc_score(true_labels, pos_probs)
                        metrics['average_precision'] = average_precision_score(true_labels, pos_probs)
                        metrics['brier_score'] = brier_score_loss(true_labels, pos_probs)
                    except Exception as e:
                        logging.warning(f"Binary metric calculation error: {str(e)}")
                        metrics['roc_auc'] = np.nan
                        metrics['average_precision'] = np.nan
                        metrics['brier_score'] = np.nan
                    
                else:
                    # --- Multi-class case ---
                    # Binarize true labels for multi-class ROC AUC and average precision
                    try:
                        true_labels_binarized = label_binarize(true_labels, classes=unique_classes)
                        
                        if all_probabilities.ndim == 2 and all_probabilities.shape[1] == n_classes:
                            metrics['roc_auc'] = roc_auc_score(
                                true_labels_binarized, all_probabilities, 
                                average='macro', multi_class='ovr'
                            )
                            metrics['average_precision'] = average_precision_score(
                                true_labels_binarized, all_probabilities, average='macro'
                            )
                            # Compute average per-class Brier score
                            brier_scores = []
                            for i in range(n_classes):
                                brier_scores.append(brier_score_loss(true_labels_binarized[:, i], all_probabilities[:, i]))
                            metrics['brier_score'] = np.mean(brier_scores)
                        else:
                            logging.warning(
                                f"Probability shape mismatch: expected {n_classes} columns, got {all_probabilities.shape[1]}"
                            )
                            metrics['roc_auc'] = np.nan
                            metrics['average_precision'] = np.nan
                            metrics['brier_score'] = np.nan
                    except Exception as e:
                        logging.warning(f"Multi-class metric calculation error: {str(e)}")
                        metrics['roc_auc'] = np.nan
                        metrics['average_precision'] = np.nan
                        metrics['brier_score'] = np.nan
                
                # --- Compute Expected Calibration Error (ECE) ---
                # Using maximum predicted probability as the confidence measure.
                prob_max = np.max(all_probabilities, axis=1)
                correct = (all_predictions == true_labels).astype(float)
                n_bins = 10
                bins = np.linspace(0, 1, n_bins + 1)
                ece = 0.0
                for i in range(n_bins):
                    bin_mask = (prob_max >= bins[i]) & (prob_max < bins[i + 1])
                    if np.any(bin_mask):
                        avg_conf = np.mean(prob_max[bin_mask])
                        avg_acc = np.mean(correct[bin_mask])
                        ece += np.abs(avg_conf - avg_acc) * np.sum(bin_mask) / len(prob_max)
                metrics['ece'] = ece
                
            except Exception as e:
                logging.error(f"Error calculating advanced metrics: {str(e)}")
                metrics['roc_auc'] = np.nan
                metrics['average_precision'] = np.nan
                metrics['brier_score'] = np.nan
                metrics['ece'] = np.nan
        
        return metrics

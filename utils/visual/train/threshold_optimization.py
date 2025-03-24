import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from typing import Tuple


def plot_threshold_optimization(
    true_labels: np.ndarray,
    probabilities: np.ndarray
) -> Tuple[plt.Figure, float]:
    """
    Plot precision, recall, and F1 score against threshold values

    Args:
        true_labels: Ground truth labels
        probabilities: Predicted probabilities

    Returns:
        Tuple of (matplotlib Figure, optimal threshold)
    """
    # Get probabilities for positive class
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        pos_probs = probabilities[:, 1]
    else:
        pos_probs = probabilities

    # Calculate precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(
        true_labels, pos_probs)

    # Calculate F1 score for each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

    # Create precision-recall vs threshold curve
    fig = plt.figure(figsize=(10, 6))

    # Plot the curves
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1 Score')

    # Find best threshold (maximum F1 score)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    # Mark optimal threshold
    plt.axvline(x=best_threshold, color='k', linestyle='-', alpha=0.3)
    plt.text(best_threshold, 0.5, f'Best threshold: {best_threshold:.4f}',
             rotation=90, verticalalignment='center')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, best_threshold


def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray
) -> plt.Figure:
    """
    Plot ROC curve for binary classification

    Args:
        true_labels: Ground truth labels
        probabilities: Predicted probabilities

    Returns:
        matplotlib Figure
    """
    # Get probabilities for positive class
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        pos_probs = probabilities[:, 1]
    else:
        pos_probs = probabilities

    # Calculate ROC curve points and AUC
    fpr, tpr, _ = roc_curve(true_labels, pos_probs)
    roc_auc = auc(fpr, tpr)

    # Create ROC curve plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def save_threshold_optimization(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    task_name: str,
    dataset_name: str,
    model_name: str
) -> float:
    """
    Save threshold optimization and ROC curve plots

    Args:
        true_labels: Ground truth labels
        probabilities: Predicted probabilities
        task_name: Name of the task
        dataset_name: Name of the dataset
        model_name: Name of the model

    Returns:
        Best threshold value
    """
    best_threshold = 0.5  # Default threshold

    try:
        # Create directory for visualizations
        viz_dir = os.path.join(
            'out', task_name, dataset_name, model_name, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Plot and save threshold optimization curve
        threshold_fig, best_threshold = plot_threshold_optimization(
            true_labels, probabilities)
        threshold_fig.savefig(os.path.join(
            viz_dir, 'threshold_optimization.png'))
        plt.close(threshold_fig)

        # Plot and save ROC curve
        roc_fig = plot_roc_curve(true_labels, probabilities)
        roc_fig.savefig(os.path.join(viz_dir, 'binary_roc_curve.png'))
        plt.close(roc_fig)

        logging.info(f"Threshold optimization curves saved to {viz_dir}")
        logging.info(f"Optimal threshold: {best_threshold:.4f}")

    except Exception as e:
        logging.error(f"Error creating threshold optimization curves: {e}")
        plt.close('all')  # Close all figures on error

    return best_threshold

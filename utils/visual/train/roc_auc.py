import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import logging

logging.basicConfig(level=logging.INFO)


def plot_roc_auc(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    unique_classes = np.unique(true_labels)
    if class_names is None:
        class_names = [str(i) for i in unique_classes]
    n_classes = len(class_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fix the predictions shape for binary classification
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        # If we have probabilities with shape (n_samples, 1), reshape to (n_samples,)
        predictions = predictions.flatten()
    
    # Handle binary classification differently
    if n_classes == 2:
        # If shape is (n_samples,), use directly for binary classification
        if predictions.ndim == 1:
            # For binary classification, use the probability as is
            fpr, tpr, _ = roc_curve(true_labels, predictions)
            auc_score = roc_auc_score(true_labels, predictions)
            ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.2f}')
        else:
            # If we have 2 columns [class0_prob, class1_prob], use the second column
            if predictions.shape[1] >= 2:
                fpr, tpr, _ = roc_curve(true_labels, predictions[:, 1])
                auc_score = roc_auc_score(true_labels, predictions[:, 1])
                ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.2f}')
            else:
                # If for some reason we only have one column in a 2D array
                fpr, tpr, _ = roc_curve(true_labels, predictions.flatten())
                auc_score = roc_auc_score(true_labels, predictions.flatten())
                ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.2f}')
    else:
        # Multi-class case
        # Ensure predictions is a 2D array with appropriate columns
        if predictions.ndim == 1:
            # Reshape to match number of classes if possible
            predictions = np.vstack([1-predictions, predictions]).T
        
        # Pad with zeros if needed
        if predictions.ndim == 2 and predictions.shape[1] < n_classes:
            missing = n_classes - predictions.shape[1]
            predictions = np.hstack([predictions, np.zeros((predictions.shape[0], missing))])
        
        # Binarize true labels for multi-class
        true_labels_bin = label_binarize(true_labels, classes=unique_classes)
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            if i < predictions.shape[1]:
                fpr, tpr, _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
                auc_score = roc_auc_score(true_labels_bin[:, i], predictions[:, i])
                ax.plot(fpr, tpr, lw=2, 
                        label=f'Class {class_names[i]} (AUC = {auc_score:.2f})')
            else:
                logging.warning(f'Missing prediction output for class index {i} in model: {model_name}')

    # Add reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_title(f"ROC Curve for {model_name}")
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    return fig


def save_roc_auc(models, true_labels_dict, predictions_dict, class_names=None, task_name='', dataset_name=''):
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    for model_name in models:
        output_dir = os.path.join(
            'out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        fig = plot_roc_auc(
            model_name, true_labels_dict[model_name], predictions_dict[model_name], class_names, dataset_name)
        fig.savefig(os.path.join(output_dir, 'roc_auc.png'),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import logging

logging.basicConfig(level=logging.INFO)


def plot_roc_auc(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    # Ensure predictions is a 2D array.
    if predictions.ndim == 1:
        # Assume binary: create two columns: [1-p, p]
        predictions = np.vstack([1 - predictions, predictions]).T
    elif predictions.ndim == 2 and predictions.shape[1] < len(np.unique(true_labels)):
        # Pad missing columns with zeros
        missing = len(np.unique(true_labels)) - predictions.shape[1]
        predictions = np.hstack(
            [predictions, np.zeros((predictions.shape[0], missing))])

    unique_classes = np.unique(true_labels)
    if class_names is None:
        class_names = [str(i) for i in unique_classes]
    n_classes = len(class_names)

    # Binarize true labels
    true_labels_bin = label_binarize(true_labels, classes=unique_classes)

    fig, ax = plt.subplots(figsize=(10, 8))

    # For each class, compute ROC curve and AUC
    for i in range(n_classes):
        if i < predictions.shape[1]:
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
            auc_score = roc_auc_score(true_labels_bin[:, i], predictions[:, i])
            ax.plot(fpr, tpr, lw=2,
                    label=f'Class {class_names[i]} (AUC = {auc_score:.2f})')
        else:
            logging.warning(
                f'Missing prediction output for class index {i} in model: {model_name}')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_title(f"ROC Curve for {model_name}")
    ax.grid(True)
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

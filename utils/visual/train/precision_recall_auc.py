import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import logging

logging.basicConfig(level=logging.INFO)


def plot_precision_recall_auc(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    # Ensure predictions is a 2D array.
    if predictions.ndim == 1:
        # Assume binary classification: create two columns: [1 - p, p]
        predictions = np.vstack([1 - predictions, predictions]).T

    # Generate default class names based on unique classes
    unique_classes = np.unique(true_labels)
    class_names = [str(i) for i in unique_classes]
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Binarize the true labels
    true_labels_bin = label_binarize(true_labels, classes=unique_classes)
    # logging.info(f'Binarized true labels shape: {true_labels_bin.shape}')

    predictions = np.array(predictions, dtype=float)

    for i in range(n_classes):
        if i < predictions.shape[1]:
            precision, recall, _ = precision_recall_curve(
                true_labels_bin[:, i], predictions[:, i])
            avg_precision = average_precision_score(
                (true_labels == i).astype(int), predictions[:, i])
            ax.plot(recall, precision, lw=2,
                    label=f'class {class_names[i]} (AP = {avg_precision:.2f})')
        else:
            logging.warning(
                f'Missing class {i} in predictions for model: {model_name}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(title="Model and Class", loc='upper right')
    ax.set_title('Precision-Recall AUC Curve')
    ax.grid(True)
    plt.tight_layout()
    return fig


def save_precision_recall_auc(models, true_labels_dict, predictions_dict, class_names=None, task_name='', dataset_name=''):
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    for model_name in models:
        output_dir = os.path.join(
            'out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        fig = plot_precision_recall_auc(
            model_name, true_labels_dict[model_name], predictions_dict[model_name], class_names, dataset_name)
        fig.savefig(os.path.join(output_dir, 'precision_recall_auc.png'))
        plt.close(fig)

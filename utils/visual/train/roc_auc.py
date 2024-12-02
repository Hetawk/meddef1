import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import logging

logging.basicConfig(level=logging.INFO)

def plot_roc_auc(model_name, true_labels, predictions, class_names=None, dataset_name=''):
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
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
            roc_auc = roc_auc_score(true_labels_bin[:, i], predictions[:, i])
            ax.plot(fpr, tpr, lw=2, label=f'{model_name} class {class_names[i]} (AUC = {roc_auc:.2f})')
        else:
            logging.warning(f'Missing class {i} in predictions for model: {model_name}')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(title="Model and Class", loc='lower right')
    ax.set_title('ROC AUC Curve')
    ax.grid(True)
    plt.tight_layout()
    return fig

def save_roc_auc(models, true_labels_dict, predictions_dict, class_names=None, task_name='', dataset_name=''):
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    for model_name in models:
        output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        fig = plot_roc_auc(model_name, true_labels_dict[model_name], predictions_dict[model_name], class_names, dataset_name)
        fig.savefig(os.path.join(output_dir, 'roc_auc.png'))
        plt.close(fig)
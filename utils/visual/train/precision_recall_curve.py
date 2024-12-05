import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def plot_precision_recall_curve(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    # Generate default class names based on unique classes
    unique_classes = np.unique(true_labels)
    class_names = [str(i) for i in unique_classes]
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8))  # Create a new figure with a specified size

    # Binarize the true labels
    true_labels_bin = label_binarize(true_labels, classes=unique_classes)
    # logging.info(f'Binarized true labels shape: {true_labels_bin.shape}')

    # Convert predictions to a NumPy array and ensure it has the correct shape
    predictions = np.array(predictions, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # Compute Precision-Recall and plot curve for each class
    for i in range(n_classes):
        if i < predictions.shape[1]:  # Check if the column exists in the predictions array
            precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], predictions[:, i])
            ax.plot(recall, precision, lw=2, label=f'{model_name} class {class_names[i]}')
            # logging.info(f'Plotted precision-recall curve for class {class_names[i]}')
        else:
            logging.warning(f'Missing class {i} in predictions for model: {model_name}')
            pass  # Skipping silently

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(title="Model and Class", loc='upper right')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True)
    plt.tight_layout()
    return fig

def save_precision_recall_curve(models, true_labels_dict, predictions_dict, class_names=None, task_name='', dataset_name=''):
    # Ensure task_name and dataset_name are strings
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    for model_name in models:
        output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        fig = plot_precision_recall_curve(model_name, true_labels_dict[model_name], predictions_dict[model_name], class_names, dataset_name)
        fig.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close(fig)

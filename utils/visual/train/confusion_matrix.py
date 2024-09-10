# confusion_matrix.py
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(model_name, true_labels, predictions, class_names):
    # Flatten the predictions list if it is nested
    if isinstance(predictions[0], list):
        predictions = [pred[0] for pred in predictions]

    # Ensure true_labels and predictions are lists of integers
    true_labels = [int(label) for label in true_labels]
    predictions = [int(pred) for pred in predictions]

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.matshow(conf_matrix, cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)  # Adjust space for class names
    return fig

def save_confusion_matrix(models, true_labels_dict, predictions_dict, class_names, task_name, dataset_name):
    # Ensure task_name and dataset_name are strings
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    for model_name in models:
        try:
            fig = plot_confusion_matrix(model_name, true_labels_dict[model_name], predictions_dict[model_name], class_names)
            fig.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
            plt.close(fig)
        except ValueError as e:
            logging.error(f"Error generating confusion matrix for {model_name}: {e}")

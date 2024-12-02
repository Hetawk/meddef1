import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    # Convert logits to predicted class indices
    if isinstance(predictions[0], list):
        predictions = np.argmax(predictions, axis=1)

    # Ensure true_labels and predictions are lists of integers
    true_labels = [int(label) for label in true_labels]
    predictions = [int(pred) for pred in predictions]

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Generate default class names based on unique classes
    unique_classes = np.union1d(true_labels, predictions)
    class_names = [str(i) for i in unique_classes]

    # Ensure the number of class names matches the number of unique classes
    if len(class_names) != len(unique_classes):
        error_message = f"Confusion Matrix: - The number of class names ({len(class_names)}) does not match the number of unique classes ({len(unique_classes)}) in the data."
        logging.error(error_message)
        print(f"Confusion Matrix: - Error: {error_message}")
        print(f"Confusion Matrix: - Class names: {class_names}")
        print(f"Confusion Matrix: - Unique classes: {unique_classes}")
        raise ValueError(error_message)

    # Debugging prints
    print(f"Confusion Matrix: - Confusion matrix: \n{conf_matrix}")
    print(f"Confusion Matrix: - Class names: {class_names}")
    print(f"Confusion Matrix: - Unique classes: {unique_classes}")

    # Plot non-normalized and normalized confusion matrices
    titles_options = [
        (f"Confusion matrix for {model_name}, without normalization", None),
        (f"Normalized confusion matrix for {model_name}", 'true'),
    ]
    figs = []
    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, values_format='.2f' if normalize else 'd')
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        figs.append(fig)
    return figs

def save_confusion_matrix(models, true_labels_dict, predictions_dict, class_names=None, task_name='', dataset_name=''):
    # Ensure task_name and dataset_name are strings
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    for model_name in models:
        output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        try:
            figs = plot_confusion_matrix(model_name, true_labels_dict[model_name], predictions_dict[model_name], class_names, dataset_name)
            for i, fig in enumerate(figs):
                fig.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}_{i}.png'))
                plt.close(fig)
        except ValueError as e:
            logging.error(f"Confusion Matrix: - Error generating confusion matrix for {model_name}: {e}")

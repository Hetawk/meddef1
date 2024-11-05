import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Mapping from long class names to shorter names for CCTS dataset
ccts_class_name_mapping = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'class 0',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'class 1',
    'normal': 'class 2',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'class 3'
}

def plot_confusion_matrix(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    # Convert logits to predicted class indices
    if isinstance(predictions[0], list):
        predictions = np.argmax(predictions, axis=1)

    # Ensure true_labels and predictions are lists of integers
    true_labels = [int(label) for label in true_labels]
    predictions = [int(pred) for pred in predictions]

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # If class_names is not provided, generate default class names
    unique_classes = np.union1d(true_labels, predictions)
    if class_names is None:
        class_names = [f'class {i}' for i in unique_classes]
    else:
        # Check if the dataset is CCTS and map the original class names to shorter names
        if dataset_name.lower() == 'ccts':
            class_names = [ccts_class_name_mapping.get(name, name) for name in class_names]
        # Check if the dataset is SCISIC and apply simple class names
        elif dataset_name.lower() == 'scisic':
            class_names = [f'class {i}' for i in unique_classes]

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
        fig, ax = plt.subplots()
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

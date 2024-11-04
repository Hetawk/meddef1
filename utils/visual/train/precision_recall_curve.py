import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# mapping from long class names to shorter names for CCTS dataset
ccts_class_name_mapping = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'class 0',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'class 1',
    'normal': 'class 2',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'class 3'
}

def plot_precision_recall_curve(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a new figure with a specified size

    # Check if the dataset is CCTS and map the original class names to shorter names
    if dataset_name.lower() == 'ccts':
        class_names = [ccts_class_name_mapping.get(name, name) for name in class_names]

    # Binarize the true labels
    true_labels_bin = label_binarize(true_labels, classes=[i for i in range(n_classes)])
    logging.info(f'Binarized true labels shape: {true_labels_bin.shape}')

    # Convert predictions to a NumPy array and ensure it has the correct shape
    predictions = np.array(predictions, dtype=float)
    # if predictions.ndim == 1 or predictions.shape[1] != n_classes:
    #     logging.warning(f'Predictions array shape {predictions.shape} is not correct, reshaping to (number of samples, {n_classes})')
    #     predictions = np.tile(predictions, (1, n_classes))
    #
    # logging.info(f'Predictions shape after reshaping: {predictions.shape}')

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

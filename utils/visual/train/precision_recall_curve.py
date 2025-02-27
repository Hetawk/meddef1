import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def plot_precision_recall_curve(model_name, true_labels, predictions, class_names=None, dataset_name=''):
    # Generate default class names based on unique classes
    unique_classes = np.unique(true_labels)
    if class_names is None:
        class_names = [str(i) for i in unique_classes]
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8))  # Create a new figure with a specified size

    # Fix the predictions shape for binary classification
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        # If we have probabilities with shape (n_samples, 1), reshape to (n_samples,)
        predictions = predictions.flatten()
    
    # Handle binary classification differently
    if n_classes == 2:
        # If shape is (n_samples,), use directly for binary classification
        if predictions.ndim == 1:
            # For binary classification, use the probability as is
            precision, recall, _ = precision_recall_curve(true_labels, predictions)
            ap_score = average_precision_score(true_labels, predictions)
            ax.plot(recall, precision, lw=2, label=f'AP = {ap_score:.2f}')
        else:
            # If we have 2 columns [class0_prob, class1_prob], use the second column
            if predictions.shape[1] >= 2:
                precision, recall, _ = precision_recall_curve(true_labels, predictions[:, 1])
                ap_score = average_precision_score(true_labels, predictions[:, 1])
                ax.plot(recall, precision, lw=2, label=f'AP = {ap_score:.2f}')
            else:
                # If for some reason we only have one column in a 2D array
                precision, recall, _ = precision_recall_curve(true_labels, predictions.flatten())
                ap_score = average_precision_score(true_labels, predictions.flatten())
                ax.plot(recall, precision, lw=2, label=f'AP = {ap_score:.2f}')
    else:
        # Multi-class case
        # Convert predictions to a NumPy array and ensure it has the correct shape
        predictions = np.array(predictions, dtype=float)
        if predictions.ndim == 1:
            # Reshape to match number of classes if possible
            predictions = np.vstack([1-predictions, predictions]).T
            
        # Pad with zeros if needed
        if predictions.ndim == 2 and predictions.shape[1] < n_classes:
            missing = n_classes - predictions.shape[1]
            predictions = np.hstack([predictions, np.zeros((predictions.shape[0], missing))])
            
        # Binarize the true labels for multi-class
        true_labels_bin = label_binarize(true_labels, classes=unique_classes)
        
        # Compute Precision-Recall and plot curve for each class
        for i in range(n_classes):
            if i < predictions.shape[1]:
                precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], predictions[:, i])
                ap_score = average_precision_score(true_labels_bin[:, i], predictions[:, i])
                ax.plot(recall, precision, lw=2, label=f'Class {class_names[i]} (AP = {ap_score:.2f})')
            else:
                logging.warning(f'Missing class {i} in predictions for model: {model_name}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='upper right')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
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

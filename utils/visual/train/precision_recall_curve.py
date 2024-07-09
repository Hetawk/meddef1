# precision_recall_curve.py
import os

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

def plot_precision_recall_curve(models, true_labels, predictions, class_names):
    n_classes = len(class_names)
    fig = plt.figure()  # Create a new figure

    # Compute Precision-Recall and plot curve for each model
    for model_name in models:
        true_labels_model, predictions_model = np.array(true_labels[model_name]), np.array(predictions[model_name])
        true_labels_bin = label_binarize(true_labels_model, classes=[i for i in range(n_classes)])

        for i in range(n_classes):
            if i < predictions_model.shape[1]:  # Check if the column exists in the predictions array
                precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], predictions_model[:, i])
                plt.plot(recall, precision, lw=2, label=f'{model_name} class {class_names[i]}')
            else:
                # Optionally log a message or handle missing classes differently
                pass  # Skipping silently

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best", title="Model and Class")
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    return fig

def save_precision_recall_curve(models, true_labels, predictions, class_names, task_name, dataset_name):
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_precision_recall_curve(models, true_labels, predictions, class_names)
    fig.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close(fig)
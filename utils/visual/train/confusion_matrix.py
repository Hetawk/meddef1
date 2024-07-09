# confusion_matrix.py

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix(true_labels, predictions, model_name, class_labels):
    true_labels, predictions = np.array(true_labels), np.array(predictions)
    cm = confusion_matrix(true_labels, predictions, labels=class_labels)

    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    return fig

def combine_confusion_matrices(models, true_labels, predictions, class_labels):
    combined_cm = None
    for model_name in models:
        cm = confusion_matrix(true_labels[model_name], predictions[model_name], labels=class_labels)
        if combined_cm is None:
            combined_cm = cm
        else:
            combined_cm += cm
    return combined_cm

def save_confusion_matrix(models, true_labels, predictions, task_name, dataset_name, class_labels):
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    for model_name in models:
        fig = plot_confusion_matrix(true_labels[model_name], predictions[model_name], model_name, class_labels)
        fig.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
        plt.close(fig)

    # Save the combined confusion matrix
    combined_cm = combine_confusion_matrices(models, true_labels, predictions, class_labels)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(combined_cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Combined Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
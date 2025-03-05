# # normal_visual.py

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     confusion_matrix, precision_recall_curve,
#     ConfusionMatrixDisplay
# )
# from sklearn.preprocessing import label_binarize
# from collections.abc import Iterable

# def flatten_list(input_list):
#     if isinstance(input_list, Iterable) and not isinstance(input_list, (str, bytes, np.ndarray)):
#         return [item for sublist in input_list for item in sublist]
#     else:
#         return [input_list] if not isinstance(input_list, list) else input_list

# # Ensure ensure_numpy_array handles numpy.int64 correctly
# def ensure_numpy_array(input_data):
#     if isinstance(input_data, list):
#         return np.array(input_data)
#     elif isinstance(input_data, np.ndarray):
#         return input_data
#     else:
#         return np.array([input_data])

# # Plotting and saving confusion matrix
# def plot_confusion_matrix(true_labels, predictions, model_name, class_labels):
#     class_labels = flatten_list(class_labels)
#     true_labels = flatten_list(true_labels)
#     predictions = flatten_list(predictions)

#     true_labels = ensure_numpy_array(true_labels)
#     predictions = ensure_numpy_array(predictions)

#     cm = confusion_matrix(true_labels, predictions, labels=class_labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     disp.plot(ax=ax, cmap='viridis')
#     plt.title(f'Confusion Matrix for {model_name}')
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     return fig

# def save_confusion_matrix(true_labels, predictions, model_name, class_labels, task_name, dataset_name):
#     fig = plot_confusion_matrix(true_labels, predictions, model_name, class_labels)
#     output_path = f'out/{task_name}/{dataset_name}/{model_name}/confusion_matrix_{model_name}.png'
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     fig.savefig(output_path)
#     plt.close(fig)
#     print(f'Confusion matrix saved to {output_path}')

# # Plotting and saving precision-recall curve
# def plot_precision_recall_curve(models, true_labels, predictions, class_names):
#     n_classes = len(class_names)
#     fig, ax = plt.subplots()

#     for model_name in models:
#         true_labels_model = np.array(true_labels[model_name])
#         predictions_model = np.array(predictions[model_name])
#         true_labels_bin = label_binarize(true_labels_model, classes=[i for i in range(n_classes)])

#         for i in range(n_classes):
#             if i < predictions_model.shape[1]:
#                 precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], predictions_model[:, i])
#                 ax.plot(recall, precision, lw=2, label=f'{model_name} class {class_names[i]}')

#     ax.set_xlabel('Recall')
#     ax.set_ylabel('Precision')
#     ax.legend(loc="best", title="Model and Class")
#     ax.set_title('Precision-Recall Curve')
#     ax.grid(True)
#     return fig

# def save_precision_recall_curve(models, true_labels, predictions, class_names, task_name, dataset_name):
#     for model_name in models:
#         output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
#         os.makedirs(output_dir, exist_ok=True)

#         fig = plot_precision_recall_curve(models, true_labels, predictions, class_names)
#         fig.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
#         plt.close(fig)

# # Plotting and saving training-validation loss/accuracy
# def plot_training_validation_loss_accuracy(history):
#     epochs = np.arange(len(history['epoch'])) + 1
#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss', color=color)
#     ax1.plot(epochs, history['loss'], label='Training Loss', color=color)
#     ax1.plot(epochs, history['val_loss'], label='Validation Loss', color=color, linestyle='dashed')
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('Accuracy', color=color)
#     ax2.plot(epochs, history['accuracy'], label='Training Accuracy', color=color)
#     ax2.plot(epochs, history['val_accuracy'], label='Validation Accuracy', color=color, linestyle='dashed')
#     ax2.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()
#     plt.title('Training and Validation Loss/Accuracy')
#     ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
#     return fig

# def save_training_validation_loss_accuracy(history, task_name, dataset_name, model_name):
#     fig = plot_training_validation_loss_accuracy(history)
#     output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
#     os.makedirs(output_dir, exist_ok=True)
#     fig_path = os.path.join(output_dir, 'training_validation_loss_accuracy.png')
#     fig.savefig(fig_path)
#     plt.close(fig)

# def load_and_visualize_training_results(task_name, dataset_name, model_name):
#     filename = os.path.join('out', task_name, dataset_name, model_name, 'training_history.csv')
#     df = pd.read_csv(filename)
#     history = {
#         'epoch': df['epoch'].tolist(),
#         'loss': df['loss'].tolist(),
#         'accuracy': df['accuracy'].tolist(),
#         'val_loss': df['val_loss'].tolist(),
#         'val_accuracy': df['val_accuracy'].tolist()
#     }
#     save_training_validation_loss_accuracy(history, task_name, dataset_name, model_name)

# # Combined method to visualize all
# def visualize_all(models, data, task_name, dataset_name, class_names):
#     true_labels_dict, predictions_dict = data
#     for model_name in models:
#         true_labels = true_labels_dict[model_name]
#         predictions = predictions_dict[model_name]
#         save_confusion_matrix(true_labels, predictions, model_name, class_names, task_name, dataset_name)

#     save_precision_recall_curve(models, true_labels_dict, predictions_dict, class_names, task_name, dataset_name)

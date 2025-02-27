import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import logging

logging.basicConfig(level=logging.INFO)

def plot_precision_recall_auc(model_names, true_labels_dict, prob_preds_dict, class_names=None, dataset_name=''):
    if not isinstance(model_names, list):
        model_names = [model_names]
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name in model_names:
        true_labels = true_labels_dict[model_name]
        predictions = prob_preds_dict[model_name]
        unique_classes = np.unique(true_labels)
        
        if class_names is None:
            class_names = [str(i) for i in unique_classes]
        n_classes = len(class_names)
        
        # Fix the predictions shape for binary classification
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        # Handle binary classification differently
        if n_classes == 2:
            # If shape is (n_samples,), use directly for binary classification
            if predictions.ndim == 1:
                precision, recall, _ = precision_recall_curve(true_labels, predictions)
                ap_score = average_precision_score(true_labels, predictions)
                ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap_score:.2f})')
            else:
                # If we have 2 columns [class0_prob, class1_prob], use the second column
                if predictions.shape[1] >= 2:
                    precision, recall, _ = precision_recall_curve(true_labels, predictions[:, 1])
                    ap_score = average_precision_score(true_labels, predictions[:, 1])
                    ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap_score:.2f})')
                else:
                    # If for some reason we only have one column in a 2D array
                    precision, recall, _ = precision_recall_curve(true_labels, predictions.flatten())
                    ap_score = average_precision_score(true_labels, predictions.flatten())
                    ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap_score:.2f})')
        else:
            # Multi-class case - here we plot the micro-average PR curve
            # Ensure predictions is a 2D array with appropriate columns
            if predictions.ndim == 1:
                predictions = np.vstack([1-predictions, predictions]).T
            
            # Pad with zeros if needed
            if predictions.ndim == 2 and predictions.shape[1] < n_classes:
                missing = n_classes - predictions.shape[1]
                predictions = np.hstack([predictions, np.zeros((predictions.shape[0], missing))])
            
            # Binarize true labels
            true_labels_bin = label_binarize(true_labels, classes=unique_classes)
            
            # Compute PR curve and AP for each class
            precision = dict()
            recall = dict()
            ap = dict()
            
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(true_labels_bin[:, i], predictions[:, i])
                ap[i] = average_precision_score(true_labels_bin[:, i], predictions[:, i])
            
            # Compute micro-average PR curve and AP
            precision_micro, recall_micro, _ = precision_recall_curve(true_labels_bin.ravel(), predictions.ravel())
            ap_micro = average_precision_score(true_labels_bin.ravel(), predictions.ravel())
            
            ax.plot(recall_micro, precision_micro, 
                 label=f'{model_name} (AP = {ap_micro:.2f})', lw=2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    
    return fig

def save_precision_recall_auc(model_names, true_labels_dict, prob_preds_dict, class_names=None, task_name='', dataset_name=''):
    task_name = str(task_name)
    dataset_name = str(dataset_name)
    
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plot_precision_recall_auc(model_names, true_labels_dict, prob_preds_dict, class_names, dataset_name)
    fig.savefig(os.path.join(output_dir, 'precision_recall_auc.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Also save to each model's directory
    for model_name in model_names if isinstance(model_names, list) else [model_names]:
        model_output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(model_output_dir, exist_ok=True)
        
        fig = plot_precision_recall_auc([model_name], true_labels_dict, prob_preds_dict, class_names, dataset_name)
        fig.savefig(os.path.join(model_output_dir, 'precision_recall_auc.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

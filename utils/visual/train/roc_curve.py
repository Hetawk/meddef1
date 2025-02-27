import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import logging

logging.basicConfig(level=logging.INFO)

def plot_roc_curve(model_names, true_labels_dict, prob_preds_dict, class_names=None, dataset_name=''):
    if not isinstance(model_names, list):
        model_names = [model_names]
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # We'll plot one model at a time
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
                fpr, tpr, _ = roc_curve(true_labels, predictions)
                auc_score = roc_auc_score(true_labels, predictions)
                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.2f})')
            else:
                # If we have 2 columns [class0_prob, class1_prob], use the second column
                if predictions.shape[1] >= 2:
                    fpr, tpr, _ = roc_curve(true_labels, predictions[:, 1])
                    auc_score = roc_auc_score(true_labels, predictions[:, 1])
                    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.2f})')
                else:
                    # If for some reason we only have one column in a 2D array
                    fpr, tpr, _ = roc_curve(true_labels, predictions.flatten())
                    auc_score = roc_auc_score(true_labels, predictions.flatten())
                    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.2f})')
        else:
            # Multi-class case - here we plot the micro-average ROC curve
            # Ensure predictions is a 2D array with appropriate columns
            if predictions.ndim == 1:
                predictions = np.vstack([1-predictions, predictions]).T
            
            # Pad with zeros if needed
            if predictions.ndim == 2 and predictions.shape[1] < n_classes:
                missing = n_classes - predictions.shape[1]
                predictions = np.hstack([predictions, np.zeros((predictions.shape[0], missing))])
            
            # Binarize true labels
            true_labels_bin = label_binarize(true_labels, classes=unique_classes)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
                roc_auc[i] = roc_auc_score(true_labels_bin[:, i], predictions[:, i])
            
            # Compute micro-average ROC curve and ROC area
            fpr_micro, tpr_micro, _ = roc_curve(true_labels_bin.ravel(), predictions.ravel())
            roc_auc_micro = roc_auc_score(true_labels_bin.ravel(), predictions.ravel())
            
            ax.plot(fpr_micro, tpr_micro, label=f'{model_name} (AUC = {roc_auc_micro:.2f})', 
                 lw=2)
    
    # Add reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    
    return fig

def save_roc_curve(model_names, true_labels_dict, prob_preds_dict, class_names=None, task_name='', dataset_name=''):
    task_name = str(task_name)
    dataset_name = str(dataset_name)
    
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plot_roc_curve(model_names, true_labels_dict, prob_preds_dict, class_names, dataset_name)
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Also save to each model's directory
    for model_name in model_names if isinstance(model_names, list) else [model_names]:
        model_output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(model_output_dir, exist_ok=True)
        
        fig = plot_roc_curve([model_name], true_labels_dict, prob_preds_dict, class_names, dataset_name)
        fig.savefig(os.path.join(model_output_dir, 'roc_curve.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

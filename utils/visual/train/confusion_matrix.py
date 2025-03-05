import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Union, Optional


def format_class_name(name: str, max_length: int = 20) -> str:
    """Format class name to be display-friendly"""
    if len(name) <= max_length:
        return name

    # For long names, try to split on common separators
    separators = ['_', '.', '-', ' ']
    for sep in separators:
        if sep in name:
            parts = name.split(sep)
            # Take first letters of each part except the last
            abbreviated = '.'.join(part[0].upper() for part in parts[:-1])
            # Keep the last part if it's short enough, otherwise truncate
            last_part = parts[-1][:max_length-len(abbreviated)-1]
            return f"{abbreviated}.{last_part}"

    # If no separators found, just truncate with ellipsis
    return f"{name[:max_length-3]}..."


def plot_confusion_matrix(
    model_name: str,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset_name: str = "",
    display_mode: str = 'full',  # 'full', 'abbreviated', or 'numeric'
    fig_size: tuple = (12, 8)
) -> List[plt.Figure]:
    """
    Plot confusion matrix with flexible display options

    Args:
        display_mode: 
            'full' - show full class names
            'abbreviated' - show abbreviated class names
            'numeric' - show numeric indices
    """
    # Ensure inputs are numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Handle class names based on display mode
    if display_mode == 'numeric':
        display_labels = [str(i) for i in range(cm.shape[0])]
    else:
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]

        if display_mode == 'abbreviated':
            display_labels = [format_class_name(name) for name in class_names]
        else:  # 'full'
            display_labels = class_names

    # Create figures
    titles_options = [
        (f"Confusion matrix for {model_name}", None),
        (f"Normalized confusion matrix for {model_name}", 'true'),
    ]

    figs = []
    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=fig_size)

        # Create ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        )

        # Plot with enhanced visibility
        disp.plot(
            cmap='Blues',
            ax=ax,
            values_format='.2f' if normalize else 'd',
            xticks_rotation=45,
            im_kw={'interpolation': 'nearest'}
        )

        # Adjust layout for readability
        plt.title(title, pad=20)

        # Add spacing for long labels
        plt.subplots_adjust(bottom=0.2)

        # Add grid for better readability
        ax.grid(False)

        # Enhance text visibility
        for text in ax.texts:
            text.set_fontsize(8)  # Adjust cell text size

        plt.tight_layout()
        figs.append(fig)

    return figs


def save_confusion_matrix(models, true_labels_dict, predictions_dict, class_names=None, task_name='', dataset_name=''):
    """Save confusion matrices with all display modes"""
    task_name = str(task_name)
    dataset_name = str(dataset_name)

    display_modes = ['numeric', 'abbreviated', 'full']

    for model_name in models:
        output_dir = os.path.join(
            'out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Generate all display modes
            for mode in display_modes:
                figs = plot_confusion_matrix(
                    model_name=model_name,
                    true_labels=true_labels_dict[model_name],
                    predictions=predictions_dict[model_name],
                    class_names=class_names,
                    dataset_name=dataset_name,
                    display_mode=mode
                )

                for i, fig in enumerate(figs):
                    fig.savefig(os.path.join(
                        output_dir, f'confusion_matrix_{model_name}_{mode}_{i}.png'),
                        bbox_inches='tight',
                        dpi=300
                    )
                    plt.close(fig)

        except Exception as e:
            logging.error(
                f"Error generating confusion matrix for {model_name}: {e}")

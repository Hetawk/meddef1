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
    # Validate inputs
    if true_labels is None or predictions is None:
        logging.warning(
            "Empty true_labels or predictions passed to plot_confusion_matrix")
        return []

    # Ensure inputs are numpy arrays and have the same shape
    true_labels = np.array(true_labels).flatten()
    predictions = np.array(predictions).flatten()

    if len(true_labels) == 0 or len(predictions) == 0:
        logging.warning("Empty arrays passed to plot_confusion_matrix")
        return []

    if len(true_labels) != len(predictions):
        logging.warning(
            f"Mismatched array lengths: true_labels ({len(true_labels)}) vs predictions ({len(predictions)})")
        return []

    # Get unique classes from both arrays to ensure consistency
    unique_classes = np.unique(np.concatenate((true_labels, predictions)))

    if len(unique_classes) <= 1:
        logging.warning("Cannot create confusion matrix with only one class")
        return []

    # Calculate confusion matrix with explicit labels to avoid issues
    try:
        cm = confusion_matrix(true_labels, predictions, labels=unique_classes)
    except Exception as e:
        logging.error(f"Error calculating confusion matrix: {e}")
        return []

    # Handle class names based on display mode
    if display_mode == 'numeric':
        display_labels = [str(i) for i in range(cm.shape[0])]
    else:
        if class_names is None or len(class_names) < len(unique_classes):
            display_labels = [f"Class {i}" for i in unique_classes]
        elif display_mode == 'abbreviated':
            display_labels = [format_class_name(
                class_names[i]) for i in unique_classes]
        else:  # 'full'
            display_labels = [class_names[i] for i in unique_classes]

    # Create figures
    titles_options = [
        (f"Confusion matrix for {model_name}", None),
        (f"Normalized confusion matrix for {model_name}", 'true'),
    ]

    figs = []
    for title, normalize in titles_options:
        try:
            fig, ax = plt.subplots(figsize=fig_size)

            # Create ConfusionMatrixDisplay directly with our data
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=display_labels
            )

            # Plot with enhanced visibility and error handling
            norm_cm = cm.astype(
                'float') / cm.sum(axis=1)[:, np.newaxis] if normalize else cm

            # Manually set vmin and vmax to avoid the "zero-size array" error
            vmin = 0
            vmax = norm_cm.max() if normalize else cm.max()
            if vmax == 0:
                vmax = 1  # Default if all values are 0

            # Plot with explicit vmin/vmax
            disp.plot(
                cmap='Blues',
                ax=ax,
                values_format='.2f' if normalize else 'd',
                xticks_rotation=45,
                im_kw={'interpolation': 'nearest', 'vmin': vmin, 'vmax': vmax}
            )

            # Adjust layout for readability
            plt.title(title, pad=20)
            plt.subplots_adjust(bottom=0.2)
            ax.grid(False)

            # Enhance text visibility
            for text in ax.texts:
                text.set_fontsize(8)

            plt.tight_layout()
            figs.append(fig)
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
            plt.close()  # Close any partial figure

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
            if model_name not in true_labels_dict or model_name not in predictions_dict:
                logging.warning(f"Missing data for model {model_name}")
                continue

            true_labels = true_labels_dict[model_name]
            predictions = predictions_dict[model_name]

            # Check if labels and predictions are valid
            if not isinstance(true_labels, (list, np.ndarray)) or not isinstance(predictions, (list, np.ndarray)):
                logging.warning(f"Invalid data types for model {model_name}")
                continue

            if len(true_labels) == 0 or len(predictions) == 0:
                logging.warning(f"Empty data for model {model_name}")
                continue

            # Generate all display modes
            for mode in display_modes:
                figs = plot_confusion_matrix(
                    model_name=model_name,
                    true_labels=true_labels,
                    predictions=predictions,
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
            plt.close('all')  # Clean up any open figures on error

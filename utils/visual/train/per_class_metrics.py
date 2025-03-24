import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, List, Any, Optional


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    metrics_to_plot: List[str] = ['precision', 'recall', 'f1'],
    epoch: Optional[int] = None
) -> plt.Figure:
    """
    Create bar chart for per-class metrics visualization

    Args:
        metrics: Dictionary with per_class metrics
        class_names: List of class names (optional)
        metrics_to_plot: Which metrics to include in the plot
        epoch: Epoch number for title (optional)

    Returns:
        matplotlib Figure object
    """
    if 'per_class' not in metrics:
        raise ValueError(
            "Input metrics dictionary must contain 'per_class' key")

    per_class = metrics['per_class']

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(per_class))
    width = 0.25

    # Plot each metric as grouped bars
    for i, metric in enumerate(metrics_to_plot):
        values = [per_class[cls][metric] for cls in per_class]
        plt.bar(x + width * (i - 1), values, width, label=metric.capitalize())

    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Score')

    # Add epoch to title if provided
    title = 'Per-Class Metrics'
    if epoch is not None:
        title += f' - Epoch {epoch+1}'
    plt.title(title)

    # Use class names on x-axis
    plt.xticks(x, per_class.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()

    return fig


def save_per_class_metrics(
    metrics: Dict[str, Any],
    task_name: str,
    dataset_name: str,
    model_name: str,
    phase: str = "train",
    epoch: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Save per-class metrics visualization

    Args:
        metrics: Dictionary containing metrics with 'per_class' key
        task_name: Name of the task
        dataset_name: Name of the dataset
        model_name: Name of the model
        phase: Phase (train/val/test)
        epoch: Current epoch number
        class_names: List of class names (optional)
    """
    if 'per_class' not in metrics:
        logging.warning("No per-class metrics found to visualize")
        return

    try:
        # Create directory for visualizations
        viz_dir = os.path.join(
            'out', task_name, dataset_name, model_name, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Plot the metrics
        fig = plot_per_class_metrics(
            metrics,
            class_names=class_names,
            epoch=epoch
        )

        # Create filename with epoch if provided
        filename = f'per_class_metrics_{phase}'
        if epoch is not None:
            filename += f'_epoch_{epoch+1}'
        filename += '.png'

        # Save figure
        fig.savefig(os.path.join(viz_dir, filename))
        plt.close(fig)

        logging.info(
            f"Saved per-class metrics visualization to {viz_dir}/{filename}")

    except Exception as e:
        logging.warning(
            f"Failed to create per-class metrics visualization: {e}")
        plt.close('all')  # Clean up any open figures on error

# perturbation_analysis.py
import os
import matplotlib.pyplot as plt

def plot_perturbation_analysis(perturbations, class_names):
    """
    Plot histograms of perturbation magnitudes for each class.

    Parameters:
    - perturbations: Dictionary with class names as keys and lists of perturbation magnitudes as values.
    - class_names: List of class names for labeling the plot.

    Returns:
    - fig: The matplotlib figure object.
    """
    fig = plt.figure()
    for class_name in class_names:
        if class_name in perturbations:
            perturbation_data = perturbations[class_name]
            plt.hist(perturbation_data, bins=50, alpha=0.5, label=class_name)
        else:
            # Log or handle missing class_name if needed
            pass

    plt.xlabel('Perturbation Magnitude')
    plt.ylabel('Frequency')
    plt.legend(loc="best")
    plt.title('Perturbation Analysis')
    plt.grid(True)
    return fig

def save_perturbation_analysis_plot(perturbations, class_names, dataset_name, task_name, model_name):
    """
    Save the perturbation analysis plot to a file.

    Parameters:
    - perturbations: Dictionary with class names as keys and lists of perturbation magnitudes as values.
    - class_names: List of class names for labeling the plot.
    - dataset_name: Name of the dataset.
    - task_name: Name of the task (for directory structure).
    - model_name: Name of the model (for directory structure).
    """
    output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_perturbation_analysis(perturbations, class_names)
    fig.savefig(os.path.join(output_dir, 'perturbation_analysis.png'))
    plt.close(fig)
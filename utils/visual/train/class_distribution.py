import os
import matplotlib.pyplot as plt
from collections import Counter

def plot_class_distribution(model_name, true_labels, class_names):
    # Ensure true_labels is a list of integers
    true_labels = [int(label) for label in true_labels]

    # Count the occurrences of each class in true_labels using Counter
    class_counts = [Counter(true_labels).get(cls, 0) for cls in range(len(class_names))]

    # Ensure class_names is a list of strings for plotting
    class_names = [str(cls) for cls in range(len(class_names))]

    fig, ax = plt.subplots()
    ax.bar(class_names, class_counts, color='skyblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(f'Class Distribution for {model_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def save_class_distribution(true_labels_dict, class_names, task_name, dataset_name):
    for model_name, true_labels in true_labels_dict.items():
        output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        fig = plot_class_distribution(model_name, true_labels, class_names)
        fig.savefig(os.path.join(output_dir, f'class_distribution_{model_name}.png'))
        plt.close(fig)

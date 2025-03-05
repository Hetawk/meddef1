import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(data, x_labels, y_labels, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, ax=ax, cmap='viridis', annot=True, fmt='.2f')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def save_heatmap(data, x_labels, y_labels, task_name, dataset_name, title, xlabel, ylabel, filename):
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_heatmap(data, x_labels, y_labels, title, xlabel, ylabel)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

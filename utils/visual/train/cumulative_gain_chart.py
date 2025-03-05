import os
import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_gain_chart(data, model_name, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(data['x'], data['y'], label=model_name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def save_cumulative_gain_chart(data, model_name, task_name, dataset_name, title, xlabel, ylabel, filename):
    output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_cumulative_gain_chart(data, model_name, title, xlabel, ylabel)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

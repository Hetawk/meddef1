import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_box_violin(data, x, y, hue, kind, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 8))
    if kind == 'box':
        sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax)
    elif kind == 'violin':
        sns.violinplot(x=x, y=y, hue=hue, data=data, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def save_box_violin(data, x, y, hue, kind, task_name, dataset_name, title, xlabel, ylabel, filename):
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_box_violin(data, x, y, hue, kind, title, xlabel, ylabel)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

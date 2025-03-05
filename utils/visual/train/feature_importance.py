import os
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(importances, feature_names, model_name, title):
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(np.array(feature_names)[indices])
    ax.set_title(title)
    plt.tight_layout()
    return fig

def save_feature_importance(importances, feature_names, model_name, task_name, dataset_name, title, filename):
    output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_feature_importance(importances, feature_names, model_name, title)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

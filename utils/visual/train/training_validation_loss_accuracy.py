# training_validation_loss_accuracy.py
import os

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_training_validation_loss_accuracy(history):
    epochs = np.arange(len(history['epoch'])) + 1
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, history['loss'], label='Training Loss', color=color)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color=color, linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, history['accuracy'], label='Training Accuracy', color=color)
    ax2.plot(epochs, history['val_accuracy'], label='Validation Accuracy', color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training and Validation Loss/Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    return fig


def save_training_validation_loss_accuracy(history, task_name, dataset_name):
    fig = plot_training_validation_loss_accuracy(history)
    output_dir = os.path.join('out', task_name, dataset_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'training_validation_loss_accuracy.png')
    fig.savefig(fig_path)
    print(f"Plot saved to {fig_path}")  # Debugging: Confirm plot is saved
    plt.close(fig)
    plt.show()  # Attempt to display the plot
    print("plt.show() was called.")  # Debugging: Confirm plt.show() is reached


def load_and_visualize_training_results(task_name, dataset_name):
    filename = os.path.join('out', task_name, dataset_name, 'training_history.csv')
    df = pd.read_csv(filename)
    history = {
        'epoch': df['epoch'].tolist(),
        'loss': df['loss'].tolist(),
        'accuracy': df['accuracy'].tolist(),
        'val_loss': df['val_loss'].tolist(),
        'val_accuracy': df['val_accuracy'].tolist()
    }
    save_training_validation_loss_accuracy(history, task_name, dataset_name)
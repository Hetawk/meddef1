import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_validation_loss_accuracy(task_name, dataset_name, model_name):
    history_file = os.path.join('out', task_name, dataset_name, model_name, 'training_history.csv')
    if not os.path.isfile(history_file):
        raise FileNotFoundError(f"Training history file not found: {history_file}")

    history_df = pd.read_csv(history_file)

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    ax[0].plot(history_df['epoch'], history_df['loss'], label='Training Loss')
    ax[0].plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history_df['epoch'], history_df['accuracy'], label='Training Accuracy')
    ax[1].plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    return fig

def save_training_validation_loss_accuracy(task_name, dataset_name, model_name):
    output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_training_validation_loss_accuracy(task_name, dataset_name, model_name)
    fig.savefig(os.path.join(output_dir, 'training_validation_loss_accuracy.png'))
    plt.close(fig)
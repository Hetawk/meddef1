import matplotlib.pyplot as plt
import os
import logging
import numpy as np


def plot_adversarial_training_curves(metrics_dict, phase):
    """Plot clean vs adversarial training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(metrics_dict[phase]['clean']) + 1)
    clean_losses = [m['loss'] for m in metrics_dict[phase]['clean']]
    clean_accs = [m['accuracy'] for m in metrics_dict[phase]['clean']]
    adv_losses = [m['loss'] for m in metrics_dict[phase]['adversarial']]
    adv_accs = [m['accuracy'] for m in metrics_dict[phase]['adversarial']]
    loss_gaps = [g['loss_gap'] for g in metrics_dict[phase]['gaps']]
    acc_gaps = [g['accuracy_gap'] for g in metrics_dict[phase]['gaps']]

    # Loss plots
    ax1.plot(epochs, clean_losses, 'b-', label='Clean')
    ax1.plot(epochs, adv_losses, 'r--', label='Adversarial')
    ax1.set_title(f'{phase.capitalize()} Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plots
    ax2.plot(epochs, clean_accs, 'b-', label='Clean')
    ax2.plot(epochs, adv_accs, 'r--', label='Adversarial')
    ax2.set_title(f'{phase.capitalize()} Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Loss gap
    ax3.plot(epochs, loss_gaps, 'g-')
    ax3.set_title('Loss Gap (Adv - Clean)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Gap')

    # Accuracy gap
    ax4.plot(epochs, acc_gaps, 'g-')
    ax4.set_title('Accuracy Gap (Clean - Adv)')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Gap')

    plt.tight_layout()
    return fig


def save_adversarial_training_curves(metrics_dict, task_name, dataset_name, model_name):
    """Save adversarial training visualization"""
    output_dir = os.path.join(
        'out', task_name, dataset_name, model_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    try:
        for phase in ['train', 'val']:
            if phase in metrics_dict:
                fig = plot_adversarial_training_curves(metrics_dict, phase)
                fig.savefig(
                    os.path.join(
                        output_dir, f'adversarial_training_curves_{phase}.png'),
                    bbox_inches='tight',
                    dpi=300
                )
                plt.close(fig)
    except Exception as e:
        logging.error(f"Error saving adversarial training curves: {e}")

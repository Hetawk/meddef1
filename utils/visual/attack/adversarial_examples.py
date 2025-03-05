import matplotlib.pyplot as plt
import os
import logging

def adversarial_examples(data, model_names):
    """
    Generates a visualization of original and adversarial examples for multiple models on a single figure.

    Args:
        data (tuple): A tuple containing original images, adversarial images, and optionally labels.
        model_names (list): List of model names corresponding to the data.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the visualizations.
    """
    # Unpack data
    if len(data) == 3:
        original_images, adversarial_images, labels = data
    elif len(data) == 2:
        original_images, adversarial_images = data
        labels = None
    else:
        logging.error("Unexpected data format. Expected a tuple of length 2 or 3.")
        return None

    num_models = len(model_names)
    num_examples = min(5, len(original_images))  # Ensure we don't exceed the number of available images
    fig, axs = plt.subplots(num_models, num_examples * 2, figsize=(20, 4 * num_models))

    # If there's only one model, axs will be 1D, so we reshape it to 2D for easier indexing
    if num_models == 1:
        axs = axs.reshape(1, -1)

    for model_idx, model_name in enumerate(model_names):
        for i in range(num_examples):
            # Original image
            original_image = original_images[i].cpu().detach().permute(1, 2, 0).numpy()
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # Normalize to [0, 1]
            axs[model_idx, 2 * i].imshow(original_image)
            axs[model_idx, 2 * i].axis('off')
            axs[model_idx, 2 * i].set_title(f'Original {labels[i]}' if labels else 'Original')

            # Adversarial image
            adversarial_image = adversarial_images[i].cpu().detach().permute(1, 2, 0).numpy()
            adversarial_image = (adversarial_image - adversarial_image.min()) / (adversarial_image.max() - adversarial_image.min())  # Normalize to [0, 1]
            axs[model_idx, 2 * i + 1].imshow(adversarial_image)
            axs[model_idx, 2 * i + 1].axis('off')
            axs[model_idx, 2 * i + 1].set_title(f'Adversarial {labels[i]}' if labels else 'Adversarial')

    plt.tight_layout()
    return fig

def save_adversarial_examples(adv_examples, model_names, task_name, dataset_name, attack_name):
    """
    Saves the generated adversarial example figure to the specified directory.

    Args:
        adv_examples (tuple): A tuple containing original and adversarial images.
        model_names (list): List of model names corresponding to the data.
        task_name (str): The task name (e.g., 'attack').
        dataset_name (str): The dataset name.
        attack_name (str): The attack name.
    """
    for model_name in model_names:
        output_dir = os.path.join('out', task_name, dataset_name, model_name, attack_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        fig = adversarial_examples(adv_examples, model_names)
        if fig is not None:
            fig.savefig(os.path.join(output_dir, 'adversarial_examples.png'))
            plt.close(fig)


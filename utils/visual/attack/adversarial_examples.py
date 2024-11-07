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
    fig, axs = plt.subplots(num_models, 10, figsize=(20, 4 * num_models))

    # If there's only one model, axs will be 1D, so we reshape it to 2D for easier indexing
    if num_models == 1:
        axs = axs.reshape(1, -1)

    for model_idx, model_name in enumerate(model_names):
        for i in range(5):
            # Display original image
            original_image = original_images[i].cpu().detach()
            logging.info(f"Original image shape at model {model_name}, index {i}: {original_image.shape}")

            if original_image.dim() == 1:
                logging.warning(f"Original image at model {model_name}, index {i} is 1-dimensional.")
                axs[model_idx, 2 * i].imshow(original_image.unsqueeze(0).numpy(), cmap='gray')
            elif original_image.dim() == 3:
                axs[model_idx, 2 * i].imshow(original_image.permute(1, 2, 0).numpy())
            elif original_image.dim() == 2:
                axs[model_idx, 2 * i].imshow(original_image.numpy(), cmap='gray')
            else:
                logging.error(f"Unexpected dimension {original_image.dim()} for original image at model {model_name}, index {i}")
                continue

            axs[model_idx, 2 * i].axis('off')
            axs[model_idx, 2 * i].set_title(f'Real {i + 1}')

            # Display adversarial image
            adversarial_image = adversarial_images[i].cpu().detach()
            logging.info(f"Adversarial image shape at model {model_name}, index {i}: {adversarial_image.shape}")

            # Reshape adversarial image if necessary
            if adversarial_image.dim() == 1:
                logging.warning(f"Adversarial image at model {model_name}, index {i} is 1-dimensional.")
                continue  # Skip this image
            elif adversarial_image.dim() == 3:
                adversarial_image = adversarial_image.permute(1, 2, 0)
            elif adversarial_image.dim() == 2:
                adversarial_image = adversarial_image.unsqueeze(0)
            else:
                logging.error(f"Unexpected dimension {adversarial_image.dim()} for adversarial image at model {model_name}, index {i}")
                continue

            axs[model_idx, 2 * i + 1].imshow(adversarial_image.numpy(), cmap='gray')
            axs[model_idx, 2 * i + 1].axis('off')
            axs[model_idx, 2 * i + 1].set_title(f'Adversarial {i + 1}')

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
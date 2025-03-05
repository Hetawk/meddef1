# perturbation_visualization.py
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def perturbation_visualization(adv_examples, model_names, max_examples=10):
    figures = []  # List to store the figures
    for model_idx, model_name in enumerate(model_names):
        # Check structure of the current adversarial example
        adv_example = adv_examples[model_idx]

        # If adv_example contains only two elements (original_images, adversarial_images), handle it.
        if len(adv_example) == 2:
            original_images, adversarial_images = adv_example
            labels = None  # No labels available
        elif len(adv_example) == 3:
            original_images, adversarial_images, labels = adv_example
        else:
            logging.error(f"Unexpected format for adv_examples at index {model_idx}")
            continue

        perturbations = [adv - orig for adv, orig in zip(adversarial_images, original_images)]

        for idx, (orig, adv, perturbation) in enumerate(zip(original_images, adversarial_images, perturbations)):
            if idx >= max_examples:
                break
            orig = orig.cpu().detach().numpy().transpose((1, 2, 0))  # Detach tensor before converting to numpy
            adv = adv.cpu().detach().numpy().transpose((1, 2, 0))  # Detach tensor before converting to numpy
            perturbation = perturbation.cpu().detach().numpy().transpose(
                (1, 2, 0))  # Detach tensor before converting to numpy

            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(orig)
            axs[0].set_title(f'{model_name} Original')
            axs[1].imshow(adv)
            axs[1].set_title(f'{model_name} Adversarial')
            axs[2].imshow(np.abs(perturbation))
            axs[2].set_title(f'{model_name} Perturbation')
            figures.append(fig)  # Add the figure to the list

    return figures  # Return the list of figures


def save_perturbation_visualization(adv_examples, model_names, task_name, dataset_name):
    for model_name in model_names:
        output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        figs = perturbation_visualization(adv_examples, model_names)
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(output_dir, f'perturbation_visualization_{i}.png'))
            plt.close(fig)  # Close the figure after saving
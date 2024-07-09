# perturbation_analysis.py

import matplotlib.pyplot as plt


def perturbation_analysis(data):
    original_perturbations, defended_perturbations = data

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axs[0, i].imshow(original_perturbations[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(defended_perturbations[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
    plt.show()
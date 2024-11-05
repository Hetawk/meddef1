# visualization.py

# visualization.py

from .attack.adversarial_examples import save_adversarial_examples
from .attack.perturbation_visualization import save_perturbation_visualization
from .train.class_distribution import save_class_distribution
from .train.confusion_matrix import save_confusion_matrix
from .train.precision_recall_curve import save_precision_recall_curve
from .defense.robustness_evaluation import save_defense_robustness_plot
from .defense.perturbation_analysis import save_perturbation_analysis_plot
from .train.training_validation_loss_accuracy import load_and_visualize_training_results
import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self):
        pass

    def visualize_normal(self, models, data, task_name, dataset_name, class_names):
        true_labels_dict = {}
        predictions_dict = {}
        # Check the length of data to adapt to the presence or absence of history
        if len(data) == 3:
            true_labels_dict, predictions_dict, history = data
            # Proceed with visualization that requires history
            load_and_visualize_training_results(task_name, dataset_name)
        elif len(data) == 2:
            true_labels_dict, predictions_dict = data

        # Ensure true_labels_dict and predictions_dict are correctly populated
        for model_name in models:
            if model_name not in true_labels_dict or model_name not in predictions_dict:
                raise ValueError(f"Missing true labels or predictions for model: {model_name}")

        # Visualization that does not require history
        save_confusion_matrix(models, true_labels_dict, predictions_dict, class_names, task_name, dataset_name)
        save_precision_recall_curve(models, true_labels_dict, predictions_dict, class_names, task_name, dataset_name)
        save_class_distribution(true_labels_dict, class_names, task_name, dataset_name)

    def visualize_attack(self, original, adversarial, labels, model_name_with_depth, task_name, dataset_name,
                         attack_name):
        adv_examples = (original, adversarial, labels)
        model_names = [model_name_with_depth]
        save_adversarial_examples(adv_examples, model_names, task_name, dataset_name, attack_name)
        save_perturbation_visualization(adv_examples, model_names, task_name, dataset_name)

    def visualize_defense(self, defenses, adv_examples_dict, robustness_results, perturbations, class_names, task_name, dataset_name):
        # Save robustness vs attack plot
        defense_names = list(defenses.keys())
        attack_names = list(adv_examples_dict.keys())
        save_defense_robustness_plot(defense_names, attack_names, robustness_results, dataset_name, task_name)

        # Save perturbation analysis plot
        save_perturbation_analysis_plot(perturbations, class_names, dataset_name, task_name)



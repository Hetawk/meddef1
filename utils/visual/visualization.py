from .attack.adversarial_examples import save_adversarial_examples
from .attack.perturbation_visualization import save_perturbation_visualization
from .train.class_distribution import save_class_distribution
from .train.confusion_matrix import save_confusion_matrix
from .train.precision_recall_curve import save_precision_recall_curve
from .train.precision_recall_auc import save_precision_recall_auc
from .train.roc_auc import save_roc_auc
from .train.roc_curve import save_roc_curve
from .train.training_validation_loss_accuracy import save_training_validation_loss_accuracy
from .train.heatmaps import save_heatmap
from .defense.robustness_evaluation import save_defense_robustness_plot
from .defense.perturbation_analysis import save_perturbation_analysis_plot
from .train.adversarial_training_curves import save_adversarial_training_curves
import matplotlib.pyplot as plt
import os
import logging


class Visualization:
    def __init__(self):
        pass

    def visualize_normal(self, model_names, data, task_name, dataset_name, class_names):
        """
        data: tuple of (true_labels_dict, discrete_preds_dict, prob_preds_dict)
        """
        try:
            true_labels_dict, discrete_preds_dict, prob_preds_dict = data
        except ValueError as ve:
            logging.error(f"Error unpacking visualization data: {ve}")
            return

        # For plots that need discrete predictions (e.g. confusion matrix, class distribution)
        try:
            save_confusion_matrix(model_names, true_labels_dict,
                                  discrete_preds_dict, class_names, task_name, dataset_name)
        except Exception as e:
            logging.error(
                f"Error generating confusion matrix for {model_names}: {e}")

        try:
            save_class_distribution(
                true_labels_dict, class_names, task_name, dataset_name)
        except Exception as e:
            logging.error(
                f"Error generating class distribution for {model_names}: {e}")

        # For plots that need continuous probabilities (e.g. ROC curves, PR curves)
        try:
            save_roc_auc(model_names, true_labels_dict,
                         prob_preds_dict, class_names, task_name, dataset_name)
        except Exception as e:
            logging.error(f"Error generating ROC AUC for {model_names}: {e}")

        try:
            save_roc_curve(model_names, true_labels_dict,
                           prob_preds_dict, class_names, task_name, dataset_name)
        except Exception as e:
            logging.error(f"Error generating ROC curve for {model_names}: {e}")

        try:
            save_precision_recall_auc(
                model_names, true_labels_dict, prob_preds_dict, class_names, task_name, dataset_name)
        except Exception as e:
            logging.error(f"Error generating PR AUC for {model_names}: {e}")

        try:
            save_precision_recall_curve(
                model_names, true_labels_dict, prob_preds_dict, class_names, task_name, dataset_name)
        except Exception as e:
            logging.error(f"Error generating PR curve for {model_names}: {e}")

    def visualize_attack(self, original, adversarial, labels, model_name_with_depth, task_name, dataset_name, attack_name):
        adv_examples = (original, adversarial, labels)
        model_names = [model_name_with_depth]
        save_adversarial_examples(
            adv_examples, model_names, task_name, dataset_name, attack_name)
        save_perturbation_visualization(
            adv_examples, model_names, task_name, dataset_name)

    def visualize_defense(self, defenses, adv_examples_dict, robustness_results, perturbations, class_names, task_name, dataset_name):
        # Save robustness vs attack plot
        defense_names = list(defenses.keys())
        attack_names = list(adv_examples_dict.keys())
        save_defense_robustness_plot(
            defense_names, attack_names, robustness_results, dataset_name, task_name)

        # Save perturbation analysis plot
        save_perturbation_analysis_plot(
            perturbations, class_names, dataset_name, task_name)

    def visualize_adversarial_training(self, metrics_dict, task_name, dataset_name, model_name):
        """Visualize adversarial training metrics"""
        try:
            save_adversarial_training_curves(
                metrics_dict, task_name, dataset_name, model_name)
        except Exception as e:
            logging.error(f"Error generating adversarial training curves: {e}")

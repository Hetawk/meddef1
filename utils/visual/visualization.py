# Import the backend configuration at the top
from .train.adversarial_training_curves import save_adversarial_training_curves
from .defense.perturbation_analysis import save_perturbation_analysis_plot
from .defense.robustness_evaluation import save_defense_robustness_plot
from .train.heatmaps import save_heatmap
from .train.training_validation_loss_accuracy import save_training_validation_loss_accuracy
from .train.roc_curve import save_roc_curve
from .train.roc_auc import save_roc_auc
from .train.precision_recall_auc import save_precision_recall_auc
from .train.precision_recall_curve import save_precision_recall_curve
from .train.confusion_matrix import save_confusion_matrix
from .train.class_distribution import save_class_distribution
from .attack.perturbation_visualization import save_perturbation_visualization
from .attack.adversarial_examples import save_adversarial_examples
# Import the new visualization modules
from .train.per_class_metrics import save_per_class_metrics
from .train.threshold_optimization import save_threshold_optimization
from datetime import datetime
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.matplotlib_config import configure_matplotlib_backend

# Configure backend before other matplotlib imports
backend = configure_matplotlib_backend()


class Visualization:
    def __init__(self):
        # Log the backend being used but without timestamp
        logging.info(f"Using matplotlib backend: {backend}")
        self.figure_handles = []  # Keep track of figure handles

    # Add close_figures method to ensure proper cleanup
    def close_figures(self):
        """Close all open matplotlib figures to prevent memory leaks"""
        try:
            for fig in self.figure_handles:
                plt.close(fig)
            self.figure_handles = []
        except Exception as e:
            logging.warning(f"Error closing matplotlib figures: {e}")

    def __del__(self):
        """Ensure figures are closed when visualization object is destroyed"""
        self.close_figures()

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

    def visualize_adversarial_training(self, metrics, task_name, dataset_name, model_name):
        """Create adversarial training visualization"""
        try:
            # Create directory structure
            save_dir = os.path.join(
                'out', task_name, dataset_name, model_name, 'visualizations')
            os.makedirs(save_dir, exist_ok=True)

            # Create visualization with proper cleanup
            fig = plt.figure(figsize=(12, 8))
            self.figure_handles.append(fig)  # Track this figure

            # Your plotting code here...

            # Save figure and close it
            plt.savefig(os.path.join(save_dir, 'adversarial_metrics.png'))
            plt.close(fig)
            self.figure_handles.remove(fig)  # Remove from tracking

        except Exception as e:
            logging.error(f"Error in adversarial visualization: {e}")
            # Close any open figures
            plt.close('all')

    # New method to visualize metrics from train.py
    def visualize_metrics(self, metrics, task_name, dataset_name, model_name, phase="train", epoch=None, class_names=None):
        """
        Create and save visualizations for metrics

        Args:
            metrics: Dictionary containing metrics
            task_name: Name of the task
            dataset_name: Name of the dataset
            model_name: Name of the model
            phase: Phase (train/val/test)
            epoch: Current epoch number (optional)
            class_names: List of class names (optional)
        """
        # Only create visualizations periodically if epoch is provided
        if epoch is not None and epoch % 10 != 0 and epoch != getattr(self, 'epochs', 0) - 1:
            return

        try:
            # Create confusion matrix if metrics include it
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                true_labels = metrics.get('true_labels', [])
                predictions = metrics.get('predictions', [])

                # Use existing confusion matrix function
                save_confusion_matrix(
                    [model_name],
                    {model_name: true_labels},
                    {model_name: predictions},
                    class_names,
                    task_name,
                    dataset_name
                )

            # Create per-class metrics visualization
            if 'per_class' in metrics:
                save_per_class_metrics(
                    metrics,
                    task_name,
                    dataset_name,
                    model_name,
                    phase=phase,
                    epoch=epoch,
                    class_names=class_names
                )

        except Exception as e:
            logging.warning(f"Failed to create visualization: {e}")
            plt.close('all')  # Close any open figures on error

    # New method for threshold optimization curves
    def create_threshold_curve(self, true_labels, probabilities, task_name, dataset_name, model_name):
        """
        Create and save threshold optimization curves for binary classification

        Args:
            true_labels: Ground truth labels
            probabilities: Predicted probabilities
            task_name: Name of the task
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            Best threshold value
        """
        return save_threshold_optimization(
            true_labels,
            probabilities,
            task_name,
            dataset_name,
            model_name
        )

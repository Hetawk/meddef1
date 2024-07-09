# visualization.py

from .attack.adversarial_examples import save_adversarial_examples
from .attack.perturbation_visualization import save_perturbation_visualization
from .train.confusion_matrix import save_confusion_matrix
from .train.precision_recall_curve import save_precision_recall_curve
from .defense.robustness_evaluation import robustness_evaluation
from .defense.perturbation_analysis import perturbation_analysis
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

        # Visualization that does not require history
        save_confusion_matrix(models, true_labels_dict, predictions_dict, task_name, dataset_name, class_names)
        save_precision_recall_curve(models, true_labels_dict, predictions_dict, class_names, task_name, dataset_name)

    def visualize_attack(self, adv_examples, model_names, task_name, dataset_name, attack_name):
        save_adversarial_examples(adv_examples, model_names, task_name, dataset_name, attack_name)
        save_perturbation_visualization(adv_examples, model_names, task_name, dataset_name)

    def visualize_defense(self, data, task_name, dataset_name):
        os.makedirs(os.path.join('out', task_name, dataset_name, 'visualization'), exist_ok=True)

        plt.figure()
        robustness_evaluation(data)
        plt.savefig(os.path.join('out', task_name, dataset_name, 'visualization', 'robustness_evaluation.png'))
        plt.close()  # Close the figure after saving

        plt.figure()
        perturbation_analysis(data)
        plt.savefig(os.path.join('out', task_name, dataset_name, 'visualization', 'perturbation_analysis.png'))
        plt.close()
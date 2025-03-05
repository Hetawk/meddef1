# robustness_evaluation.py

import os
import matplotlib.pyplot as plt

def plot_robustness_vs_attack(defense_names, attack_names, robustness_results, dataset_name):
    fig = plt.figure()
    for defense_name in defense_names:
        robustness_scores = robustness_results[defense_name]
        for attack_name in attack_names:
            if attack_name in robustness_scores:
                plt.plot(robustness_scores[attack_name], label=f'{defense_name} - {attack_name}')

    plt.xlabel('Attack Type')
    plt.ylabel('Robustness Score')
    plt.legend(loc="best")
    plt.title('Robustness vs Attack')
    plt.grid(True)
    return fig

def save_defense_robustness_plot(defense_names, attack_names, robustness_results, dataset_name, task_name, model_name):
    output_dir = os.path.join('out', task_name, dataset_name, model_name, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_robustness_vs_attack(defense_names, attack_names, robustness_results, dataset_name)
    fig.savefig(os.path.join(output_dir, 'robustness_vs_attack.png'))
    plt.close(fig)
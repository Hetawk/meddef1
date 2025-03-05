import numpy as np
import torch
from sklearn.metrics import accuracy_score
import logging
from utils.metrics import Metrics


def compute_adv_success_rate(true_labels: np.ndarray, clean_predictions: np.ndarray, adv_predictions: np.ndarray) -> float:
    """Compute attack success rate as percentage drop from clean accuracy."""
    clean_acc = accuracy_score(true_labels, clean_predictions)
    adv_acc = accuracy_score(true_labels, adv_predictions)
    if clean_acc <= 0:
        return 0.0
    return max(0, (clean_acc - adv_acc) / clean_acc * 100)


def compute_robustness_curve(model, data, true_labels, epsilons: list, config: dict = None) -> dict:
    """
    Compute accuracy vs. epsilon given a model.
    This function uses an FGSM attack as an example.
    """
    device = next(model.parameters()).device
    accuracies = {}
    from gan.attack.fgsm import fgsm_attack  # ensure FGSM is implemented
    for eps in epsilons:
        adv_data = fgsm_attack(model, data, true_labels, eps, device)
        with torch.no_grad():
            output = model(adv_data)
            preds = output.argmax(dim=1).cpu().numpy()
        acc = np.mean(preds == true_labels)
        accuracies[eps] = acc
    return accuracies


def compute_gradient_norms(model, data, target):
    """Compute average gradient norm for the given data."""
    model.zero_grad()
    data.requires_grad = True
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    grad_norms = data.grad.detach().view(data.shape[0], -1).norm(p=2, dim=1)
    return grad_norms.mean().item()


def compute_lipschitz_estimate(model, data, target):
    """Estimate local Lipschitz constant using gradients."""
    return compute_gradient_norms(model, data, target)


def calculate_adv_metrics(true_labels: np.ndarray,
                          clean_predictions: np.ndarray,
                          adv_predictions: np.ndarray = None,
                          model: torch.nn.Module = None,
                          data: torch.Tensor = None,
                          target: torch.Tensor = None,
                          config: dict = None) -> dict:
    """
    Consolidated advanced metrics.
    If adv_predictions is provided, computes adversarial success rate.
    If model, data, and target are provided, computes robustness curve and gradient norms.
    """
    adv_metrics = {}
    if adv_predictions is not None:
        adv_metrics['adv_success_rate'] = compute_adv_success_rate(
            true_labels, clean_predictions, adv_predictions)
    else:
        adv_metrics['adv_success_rate'] = None

    epsilons = config.get("epsilons", [0.01, 0.03, 0.05, 0.1, 0.15, 0.3]) if config else [
        0.01, 0.03, 0.05, 0.1, 0.15, 0.3]
    if model is not None and data is not None:
        adv_metrics['robustness_curve'] = compute_robustness_curve(
            model, data, true_labels, epsilons, config)
    else:
        adv_metrics['robustness_curve'] = None

    if model is not None and data is not None and target is not None:
        adv_metrics['gradient_norm'] = compute_gradient_norms(
            model, data, target)
        adv_metrics['lipschitz_estimate'] = compute_lipschitz_estimate(
            model, data, target)
    else:
        adv_metrics['gradient_norm'] = None
        adv_metrics['lipschitz_estimate'] = None

    # Optionally, you can add more advanced metrics here.
    return adv_metrics


class AdversarialMetrics:
    def __init__(self):
        self.metrics = {}

    def compute_performance_gap(self, clean_metrics, adv_metrics):
        """Compute gap between clean and adversarial performance"""
        gaps = {
            'loss_gap': adv_metrics['loss'] - clean_metrics['loss'],
            'accuracy_gap': clean_metrics['accuracy'] - adv_metrics['accuracy']
        }
        return gaps

    def update_adversarial_comparison(self, phase, clean_loss, clean_acc, adv_loss, adv_acc):
        """Track clean vs adversarial metrics for a phase"""
        if phase not in self.metrics:
            self.metrics[phase] = {'clean': [], 'adversarial': [], 'gaps': []}

        self.metrics[phase]['clean'].append({
            'loss': float(clean_loss),
            'accuracy': float(clean_acc)
        })

        self.metrics[phase]['adversarial'].append({
            'loss': float(adv_loss),
            'accuracy': float(adv_acc)
        })

        gaps = self.compute_performance_gap(
            self.metrics[phase]['clean'][-1],
            self.metrics[phase]['adversarial'][-1]
        )
        self.metrics[phase]['gaps'].append(gaps)

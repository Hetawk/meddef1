import argparse
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader
from gan.attack.attack_loader import AttackLoader


def evaluate_at_multiple_epsilons(model, test_loader, attack_type, device, epsilons=None):
    """Test model robustness across different perturbation magnitudes"""
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

    results = {}
    attack_config = argparse.Namespace(
        attack_type=attack_type,
        attack_steps=20,
        attack_alpha=0.01,
        epsilon=0.0  # Will be overridden in the loop
    )

    attack_loader = AttackLoader(model, attack_config)

    # Test with clean examples first
    clean_acc = evaluate_clean(model, test_loader, device)
    results[0.0] = clean_acc
    logging.info(f"Clean accuracy: {clean_acc:.4f}")

    # Test with adversarial examples at each epsilon
    for eps in epsilons[1:]:  # Skip 0.0 as we already tested clean accuracy
        attack_config.epsilon = eps
        attack = attack_loader.get_attack(attack_type)

        correct = 0
        total = 0

        for data, target in tqdm(test_loader, desc=f"Testing ε={eps:.3f}"):
            data, target = data.to(device), target.to(device)

            # Generate adversarial examples
            if hasattr(attack, 'generate'):
                adv_data = attack.generate(data, target, eps)
            else:
                _, adv_data, _ = attack.attack(data, target)

            # Evaluate
            with torch.no_grad():
                output = model(adv_data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total if total > 0 else 0
        results[eps] = accuracy
        logging.info(f"Epsilon {eps:.3f}: Accuracy = {accuracy:.4f}")

    return results


def evaluate_clean(model, test_loader, device):
    """Evaluate model on clean examples"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total if total > 0 else 0


def plot_robustness_curve(results, model_name, attack_type, save_path=None):
    """Plot accuracy vs epsilon curve"""
    epsilons = sorted(list(results.keys()))
    accuracies = [results[eps] for eps in epsilons]

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, 'bo-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Perturbation Size (ε)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Robustness Curve: {model_name} vs {attack_type}', fontsize=16)

    # Add data points with values
    for eps, acc in zip(epsilons, accuracies):
        plt.annotate(f'{acc:.3f}', (eps, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved robustness curve to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model robustness across multiple epsilon values')
    parser.add_argument('--model_path', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--arch', type=str, required=True,
                        help='Model architecture')
    parser.add_argument('--depth', type=str, required=True, help='Model depth')
    parser.add_argument('--attack_type', type=str, default='pgd', choices=['fgsm', 'pgd', 'bim'],
                        help='Attack type for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset_loader = DatasetLoader()
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data,
        batch_size={'test': args.batch_size},
        num_workers=4,
        pin_memory=True
    )

    # Get number of classes
    if hasattr(test_loader.dataset, 'classes'):
        num_classes = len(test_loader.dataset.classes)
    elif hasattr(test_loader.dataset, 'class_to_idx'):
        num_classes = len(test_loader.dataset.class_to_idx)
    else:
        num_classes = 2  # Default to binary classification

    # Load model
    model_loader = ModelLoader(device, args.arch)
    model = model_loader.load_pretrained_model(
        model_name=args.arch,
        load_task='normal_training',  # Assuming this is the task name used
        dataset_name=args.data,
        depth=eval(args.depth),
        num_classes=num_classes
    )

    # Custom checkpoint loading if path provided
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded model from {args.model_path}")

    model.to(device)
    model.eval()

    # Define epsilons to test
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

    # Evaluate model
    results = evaluate_at_multiple_epsilons(
        model=model,
        test_loader=test_loader,
        attack_type=args.attack_type,
        device=device,
        epsilons=epsilons
    )

    # Plot and save results
    save_path = os.path.join('out', 'robustness_evaluation', args.data,
                             f"{args.arch}_{args.depth}_{args.attack_type}_robustness.png")
    plot_robustness_curve(
        results, f"{args.arch}_{args.depth}", args.attack_type, save_path)

    # Save numerical results
    results_path = os.path.join('out', 'robustness_evaluation', args.data,
                                f"{args.arch}_{args.depth}_{args.attack_type}_results.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"Model: {args.arch}_{args.depth}\n")
        f.write(f"Attack: {args.attack_type}\n")
        f.write("Epsilon,Accuracy\n")
        for eps, acc in sorted(results.items()):
            f.write(f"{eps:.3f},{acc:.4f}\n")

    logging.info(f"Saved numerical results to {results_path}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    main()

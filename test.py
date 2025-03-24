import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Any, Optional

from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader
from utils.visual.visualization import Visualization
# Add metrics utilities for per-class metrics
from utils.metrics import Metrics
from utils.evaluator import Evaluator
# Import the centralized argument parser
from argument_parser import parse_args

# Setup logging - Fix the format string error and remove timestamp
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(args, num_classes):
    """Load a model with specified architecture"""
    model_loader = ModelLoader(
        args.device, args.arch,
        pretrained=False,  # We're loading a trained model
        fp16=args.fp16 if hasattr(args, 'fp16') else False
    )

    # Get the first model (we usually only test one at a time)
    models_and_names = model_loader.get_model(
        model_name=args.arch[0],
        depth=args.depth,
        input_channels=3,
        num_classes=num_classes,
        task_name=args.task_name,
        dataset_name=args.data[0]
    )

    # Return the first model and its name
    model, model_name = models_and_names[0]

    # Load the trained weights
    state_dict = torch.load(args.model_path, map_location=args.device)

    # Handle weight_orig and weight_mask from pruned models
    new_state = {}
    for key, value in state_dict.items():
        if key.endswith('weight_orig'):
            new_key = key[:-len('_orig')]
            new_state[new_key] = value
        elif key.endswith('weight_mask'):
            continue
        else:
            new_state[key] = value

    model.load_state_dict(new_state)
    model.to(args.device)
    model.eval()

    return model, model_name


def test_model(model, test_loader, args):
    """Test the model on the provided test loader"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []  # Add collection of probabilities

    # For adversarial testing
    if args.adversarial:
        from gan.defense.adv_train import AdversarialTraining
        adv_trainer = AdversarialTraining(
            model, torch.nn.CrossEntropyLoss(), args)

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data = data.to(args.device)
            target = target.to(args.device)

            with autocast(enabled=args.fp16 if hasattr(args, 'fp16') else False):
                output = model(data)

            # Get predictions and probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Handle adversarial testing if enabled
            if args.adversarial and args.evaluate_robustness:
                with torch.enable_grad():
                    for attack_name in args.attack_type:
                        args.attack_name = attack_name
                        # Generate adversarial examples
                        if hasattr(adv_trainer.attack, 'generate'):
                            adv_data = adv_trainer.attack.generate(
                                data, target, args.attack_eps)
                        else:
                            _, adv_data, _ = adv_trainer.attack.attack(
                                data, target)

                        # Test on adversarial examples
                        adv_output = model(adv_data)
                        adv_pred = adv_output.argmax(dim=1)

                        # Calculate adversarial accuracy
                        adv_correct = (adv_pred == target).sum().item()
                        adv_accuracy = adv_correct / target.size(0)

                        logger.info(
                            f"Adversarial accuracy ({attack_name}): {adv_accuracy:.4f}")

    return np.array(all_preds).flatten(), np.array(all_targets), np.array(all_probs)


def generate_detailed_metrics(model_name: str,
                              all_preds: np.ndarray,
                              all_targets: np.ndarray,
                              all_probs: np.ndarray,
                              class_names: List[str],
                              args) -> Dict[str, Any]:
    """
    Generate comprehensive detailed metrics for testing

    Args:
        model_name: Name of the model
        all_preds: Predicted class indices
        all_targets: Ground truth labels
        all_probs: Predicted probabilities
        class_names: List of class names
        args: Command line arguments

    Returns:
        Dictionary containing detailed metrics
    """
    # Import necessary metrics functions directly
    from sklearn.metrics import precision_score, recall_score, f1_score

    logger.info("Generating detailed metrics and visualizations...")

    # Calculate all possible metrics using the Metrics utility
    detailed_metrics = Metrics.calculate_metrics(
        all_targets, all_preds, all_probs)

    # Calculate per-class metrics
    per_class = {}
    classes = np.unique(all_targets)

    for cls in classes:
        cls_mask = (all_targets == cls)
        if sum(cls_mask) == 0:
            continue

        # Binary classification for this class (one-vs-rest)
        cls_true = (all_targets == cls).astype(int)
        cls_pred = (all_preds == cls).astype(int)

        # Calculate comprehensive metrics directly with imported functions
        try:
            cls_metrics = {
                'accuracy': np.mean(cls_true == cls_pred),
                'precision': precision_score(cls_true, cls_pred, zero_division=0),
                'recall': recall_score(cls_true, cls_pred, zero_division=0),
                'f1': f1_score(cls_true, cls_pred, zero_division=0),
                'specificity': Metrics.calculate_metrics(cls_true, cls_pred).get('specificity', 0),
                'support': np.sum(cls_true)
            }

            # Add advanced metrics when probabilities are available
            if all_probs is not None and all_probs.shape[1] > 1:
                try:
                    # For multi-class, get probability for this specific class
                    cls_probs = all_probs[:, cls]
                    # Calculate AUC and average precision
                    cls_metrics['roc_auc'] = Metrics.calculate_metrics(
                        cls_true, cls_pred, cls_probs).get('roc_auc', 0)
                    cls_metrics['avg_precision'] = Metrics.calculate_metrics(
                        cls_true, cls_pred, cls_probs).get('average_precision', 0)
                except Exception as e:
                    logger.warning(
                        f"Could not calculate advanced metrics for class {cls}: {e}")
        except Exception as e:
            logger.warning(f"Error calculating metrics for class {cls}: {e}")
            # Create partial metrics if calculation fails
            cls_metrics = {
                'accuracy': np.mean(cls_true == cls_pred),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': np.sum(cls_true)
            }

        class_name = class_names[cls] if cls < len(
            class_names) else f"Class {cls}"
        per_class[class_name] = cls_metrics

    detailed_metrics['per_class'] = per_class

    # Log per-class metrics with more detail than during training
    logger.info("Per-Class Metrics (Detailed):")
    for cls_name, metrics in per_class.items():
        base_metrics = (f"  {cls_name}: F1={metrics['f1']:.4f}, "
                        f"Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, "
                        f"Support={metrics['support']}")

        # Add AUC if available
        if 'roc_auc' in metrics:
            base_metrics += f", AUC={metrics['roc_auc']:.4f}"

        logger.info(base_metrics)

    # Create all possible visualizations using the Visualization class
    visualization = Visualization()

    # Create confusion matrix
    visualization.visualize_metrics(
        metrics=detailed_metrics,
        task_name=args.task_name,
        dataset_name=args.data[0],
        model_name=model_name,
        phase="test",
        class_names=class_names
    )

    # For multi-class, visualize all pairwise ROC curves
    if len(classes) > 2:
        logger.info("Creating multi-class ROC curves...")
        # Use visualize_normal which handles multi-class ROC curves
        visualization.visualize_normal(
            model_names=[model_name],
            data=(
                {model_name: all_targets},  # true labels dict
                {model_name: all_preds},    # predictions dict
                {model_name: all_probs}     # probabilities dict
            ),
            task_name=args.task_name,
            dataset_name=args.data[0],
            class_names=class_names
        )
    # For binary classification, create threshold optimization curve
    elif len(classes) == 2 and all_probs.shape[1] >= 2:
        logger.info(
            "Creating binary classification threshold optimization curves...")
        optimal_threshold = visualization.create_threshold_curve(
            true_labels=all_targets,
            probabilities=all_probs,
            task_name=args.task_name,
            dataset_name=args.data[0],
            model_name=model_name
        )
        logger.info(
            f"Optimal threshold for binary classification: {optimal_threshold:.4f}")

    return detailed_metrics


def main():
    # Parse arguments with mode='test'
    args = parse_args(mode='test')

    # Load the dataset
    dataset_loader = DatasetLoader()
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data[0],
        batch_size={'train': 1, 'val': 1, 'test': args.batch_size},
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Get number of classes
    dataset = test_loader.dataset
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
        class_names = dataset.classes
    elif hasattr(dataset, 'class_to_idx'):
        num_classes = len(dataset.class_to_idx)
        class_names = list(dataset.class_to_idx.keys())
    else:
        raise AttributeError("Dataset does not contain class information")

    # Load the model
    model, model_name = load_model(args, num_classes)

    # Test the model
    all_preds, all_targets, all_probs = test_model(model, test_loader, args)

    # Calculate basic metrics
    accuracy = (all_preds == all_targets).mean()
    logger.info(f"Test accuracy: {accuracy:.4f}")

    # Generate classification report
    report = classification_report(
        all_targets, all_preds, target_names=class_names)
    logger.info("Classification Report:\n" + report)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Create output directory for model-specific results
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Generate detailed per-class metrics and visualizations
    detailed_metrics = generate_detailed_metrics(
        model_name, all_preds, all_targets, all_probs, class_names, args
    )

    # Create evaluator and save metrics to CSV
    evaluator = Evaluator(
        model_name=model_name,
        results=[],
        true_labels=all_targets,
        all_predictions=all_preds,
        task_name=args.task_name,
        all_probabilities=all_probs
    )
    evaluator.save_metrics(detailed_metrics, args.data[0])

    # Save predictions if requested
    if args.save_predictions:
        pred_df = pd.DataFrame({
            'true': all_targets,
            'pred': all_preds
        })
        # Add probability columns for each class
        for i, class_name in enumerate(class_names):
            pred_df[f'prob_{class_name}'] = all_probs[:, i]

        pred_df.to_csv(os.path.join(model_output_dir,
                       f"predictions.csv"), index=False)

    # Use visualization class for confusion matrix
    visualization = Visualization()
    visualization.visualize_normal(
        model_names=[model_name],
        data=(
            {model_name: all_targets},  # true labels dict
            {model_name: all_preds},    # predictions dict
            {model_name: all_probs}     # probabilities dict
        ),
        task_name=args.task_name,
        dataset_name=args.data[0],
        class_names=class_names
    )

    # Additional visualizations using your Visualization class
    if len(np.unique(all_targets)) == 2:  # Binary classification
        logger.info("Creating ROC and PR curves for binary classification")

        # These are already handled by visualize_normal above
        # But you can add specific visualizations here if needed

    logger.info(f"Testing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

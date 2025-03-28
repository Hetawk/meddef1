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


def test_single_image(model, image_path, class_names, device, args):
    """Test model on a single image"""
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    logger.info(f"Testing single image: {image_path}")

    # Check if file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return

    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')

        # Create standard preprocessing transform
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Preprocess image
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        # Create a non-normalized version for display
        display_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        display_tensor = display_transform(img)

        # Set model to evaluation mode
        model.eval()

        # Generate adversarial version if requested
        adv_input = None
        if args.adversarial:
            from gan.defense.adv_train import AdversarialTraining
            # Create dummy targets (this is just for visualization)
            dummy_target = torch.tensor([0], device=device)
            adv_trainer = AdversarialTraining(
                model, torch.nn.CrossEntropyLoss(), args)

            with torch.enable_grad():
                input_batch.requires_grad_(True)
                _, adv_batch, _ = adv_trainer.attack.attack(
                    input_batch, dummy_target)
                adv_input = adv_batch.detach()

                # Create perturbation visualization
                perturbation = adv_batch - input_batch
                # Scale for visibility
                enhanced_pert = (perturbation * 10) + 0.5
                enhanced_pert = torch.clamp(enhanced_pert, 0, 1)

        # Forward pass
        with torch.no_grad():
            output = model(input_batch)

            # Get adversarial prediction if available
            adv_output = model(adv_input) if adv_input is not None else None

        # Get predictions
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        adv_probs = torch.nn.functional.softmax(adv_output, dim=1)[
            0] if adv_output is not None else None

        # Get top prediction
        top_prob, top_class = torch.topk(probabilities, 1)
        predicted_class = class_names[top_class.item()]
        confidence = top_prob.item()

        # Get adversarial prediction if available
        adv_class = None
        if adv_probs is not None:
            adv_top_prob, adv_top_class = torch.topk(adv_probs, 1)
            adv_class = class_names[adv_top_class.item()]
            adv_confidence = adv_top_prob.item()

        # Print results
        logger.info(
            f"Prediction: {predicted_class} with {confidence:.4f} confidence")
        if adv_class is not None:
            logger.info(
                f"Adversarial prediction: {adv_class} with {adv_confidence:.4f} confidence")

        # Create output directory for visualizations
        os.makedirs('out/single_image_tests', exist_ok=True)

        # Create output image with prediction overlay
        fig, axs = plt.subplots(
            1, 3 if adv_input is not None else 1, figsize=(15, 5))
        if adv_input is None:
            axs = [axs]  # Make axs a list for consistent indexing

        # Display original image with prediction
        img_np = display_tensor.cpu().numpy().transpose(1, 2, 0)
        axs[0].imshow(img_np)
        axs[0].set_title(
            f"Prediction: {predicted_class}\nConfidence: {confidence:.4f}")
        axs[0].axis('off')

        # If we have adversarial version, show it too
        if adv_input is not None:
            # Display adversarial image
            adv_np = adv_input[0].cpu().detach().numpy(
            ).transpose(1, 2, 0)  # Added detach()
            # Normalize for display
            adv_np = (adv_np - adv_np.min()) / \
                (adv_np.max() - adv_np.min() + 1e-8)
            axs[1].imshow(adv_np)
            axs[1].set_title(
                f"Adversarial: {adv_class}\nConfidence: {adv_confidence:.4f}")
            axs[1].axis('off')

            # Display perturbation - Fixed the detach() issue here
            pert_np = enhanced_pert[0].cpu().detach(
            ).numpy().transpose(1, 2, 0)  # Added detach()
            axs[2].imshow(pert_np)
            axs[2].set_title(f"Perturbation (Enhanced)")
            axs[2].axis('off')

        # Save figure
        output_path = os.path.join(
            'out/single_image_tests', os.path.basename(image_path))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        # Generate class activation map (heatmap) if requested
        if args.show_heatmap:
            generate_heatmap(model, input_tensor, img, output_path, device)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error testing single image: {e}")
        import traceback
        traceback.print_exc()


def generate_heatmap(model, input_tensor, original_img, output_path, device):
    """Generate a class activation heatmap for network visualization"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        # Create GradCAM-like visualization
        # Find the last convolutional layer
        last_conv_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
                last_conv_name = name
                break

        if last_conv_layer is None:
            logger.warning(
                "Could not find convolutional layer for heatmap generation")
            return

        # Define hook to get feature maps
        feature_maps = None
        gradients = None

        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        # Register hooks
        forward_handle = last_conv_layer.register_forward_hook(forward_hook)
        backward_handle = last_conv_layer.register_full_backward_hook(
            backward_hook)

        # Set model to eval mode
        model.eval()

        # Forward pass with gradient calculation
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        pred_class = output.argmax().item()

        # Zero all gradients
        model.zero_grad()

        # Backward pass for the predicted class
        output[0, pred_class].backward()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Generate heatmap
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * feature_maps,
                            dim=1).squeeze().cpu().numpy()

        # Make positive values only and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)

        # Resize heatmap to match original image size
        heatmap = cv2.resize(
            heatmap, (original_img.width, original_img.height))

        # Convert heatmap to RGB colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert original image to numpy array
        original_np = np.array(original_img)

        # Convert RGB to BGR for OpenCV
        original_np = original_np[:, :, ::-1]

        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

        # Save heatmap visualization
        heatmap_path = output_path.replace('.png', '_heatmap.png')
        # Convert back to RGB
        cv2.imwrite(heatmap_path, superimposed[:, :, ::-1])

        logger.info(f"Heatmap saved to {heatmap_path}")

    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")


def main():
    # Parse arguments - use the unified parser instead of mode-specific one
    args = parse_args()

    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For single image testing
    if args.image_path:
        # Load dataset just to get class names
        dataset_loader = DatasetLoader()
        # Fix: Pass all required batch size keys
        _, _, test_loader = dataset_loader.load_data(
            dataset_name=args.data[0],
            # Pass all required keys
            batch_size={'train': 1, 'val': 1, 'test': 1},
            num_workers=1,
            pin_memory=False
        )

        # Get class names
        if hasattr(test_loader.dataset, 'classes'):
            class_names = test_loader.dataset.classes
        elif hasattr(test_loader.dataset, 'class_to_idx'):
            class_names = list(test_loader.dataset.class_to_idx.keys())
        else:
            class_names = ["Class 0", "Class 1"]  # Default class names

        # Get number of classes
        num_classes = len(class_names)

        # Load model
        model, model_name = load_model(args, num_classes)

        # Test single image
        test_single_image(model, args.image_path, class_names, device, args)
        return

    # Continue with regular dataset testing
    # Load the dataset with appropriate batch sizes
    dataset_loader = DatasetLoader()
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data[0],
        batch_size={'train': args.batch_size,
                    'val': args.batch_size, 'test': args.batch_size},
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if hasattr(args, 'pin_memory') else False
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

    # Use standard directory structure instead of test_results
    output_dir = os.path.join('out', args.task_name, args.data[0], model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

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

        pred_df.to_csv(os.path.join(
            output_dir, f"predictions.csv"), index=False)

    # Enhanced visualizations section using existing Visualization class
    visualization = Visualization()

    try:
        # Basic visualizations with your existing method
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

        # Add class distribution visualization from your imported class
        from utils.visual.train.class_distribution import save_class_distribution
        save_class_distribution(
            {model_name: all_targets},
            class_names,
            args.task_name,
            args.data[0]
        )
        logger.info("Generated class distribution visualization")

        # Add ROC and precision-recall curves
        from utils.visual.train.roc_curve import save_roc_curve
        from utils.visual.train.precision_recall_curve import save_precision_recall_curve

        save_roc_curve(
            [model_name],
            {model_name: all_targets},
            {model_name: all_probs},
            class_names,
            args.task_name,
            args.data[0]
        )

        save_precision_recall_curve(
            [model_name],
            {model_name: all_targets},
            {model_name: all_probs},
            class_names,
            args.task_name,
            args.data[0]
        )

        # For binary classification, also show threshold optimization
        if len(np.unique(all_targets)) == 2:
            from utils.visual.train.threshold_optimization import save_threshold_optimization

            optimal_thresh = save_threshold_optimization(
                all_targets,
                all_probs,
                args.task_name,
                args.data[0],
                model_name
            )
            logger.info(f"Optimal threshold: {optimal_thresh:.4f}")

        # If adversarial testing is enabled
        if args.adversarial and args.evaluate_robustness:
            # Create adversarial examples for visualization
            from utils.visual.attack.perturbation_visualization import save_perturbation_visualization
            from utils.visual.attack.adversarial_examples import save_adversarial_examples
            from gan.defense.adv_train import AdversarialTraining
            from itertools import islice

            # Create adversarial trainer
            adv_trainer = AdversarialTraining(
                model, torch.nn.CrossEntropyLoss(), args
            )

            # Generate and visualize adversarial examples
            vis_data, vis_targets = next(iter(test_loader))
            vis_data = vis_data[:min(5, len(vis_data))].to(device)
            vis_targets = vis_targets[:min(5, len(vis_targets))].to(device)

            with torch.enable_grad():
                original_data = vis_data.clone()
                _, adv_data, _ = adv_trainer.attack.attack(
                    original_data, vis_targets)

            adv_examples = (original_data.detach().cpu(),
                            adv_data.detach().cpu(), vis_targets.cpu())

            attack_name = args.attack_type[0] if isinstance(
                args.attack_type, list) else args.attack_type

            # Use your existing visualization methods
            save_adversarial_examples(
                adv_examples,
                [model_name],
                args.task_name,
                args.data[0],
                attack_name
            )

            save_perturbation_visualization(
                adv_examples,
                [model_name],
                args.task_name,
                args.data[0]
            )

            # Generate robustness curve
            from utils.visual.defense.robustness_evaluation import save_defense_robustness_plot

            # Test model at multiple epsilon values for robustness curve
            epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
            robustness_results = {}

            # Get clean accuracy first
            correct = 0
            total = 0
            test_subset = list(islice(test_loader, 5))

            with torch.no_grad():
                for data, target in test_subset:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)

            robustness_results[0.0] = correct / total if total > 0 else 0

            # Test with adversarial examples at each epsilon
            for eps in epsilons:
                args.attack_eps = eps
                adv_trainer = AdversarialTraining(
                    model, torch.nn.CrossEntropyLoss(), args
                )

                correct = 0
                total = 0

                for data, target in test_subset:
                    data, target = data.to(device), target.to(device)

                    with torch.enable_grad():
                        _, adv_data, _ = adv_trainer.attack.attack(
                            data, target)

                    with torch.no_grad():
                        output = model(adv_data)
                        pred = output.argmax(dim=1)
                        correct += (pred == target).sum().item()
                        total += target.size(0)

                robustness_results[eps] = correct / total if total > 0 else 0

            # Plot and save robustness curve using your visualization method
            save_defense_robustness_plot(
                [model_name],
                [attack_name],
                {"robustness": {attack_name: robustness_results}},
                args.data[0],
                args.task_name
            )

    except Exception as e:
        logger.warning(f"Error generating additional visualizations: {e}")
        import traceback
        traceback.print_exc()

    logger.info(f"Testing complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

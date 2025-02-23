import argparse
import os
import torch
from loader.dataset_loader import DatasetLoader
from train import Trainer, TrainingManager  # reuse train.py code
import json
from PIL import Image
import torch.nn.functional as F
from utils.evaluator import Evaluator
from utils.visual.visualization import Visualization
import numpy as np
from utils.adv_metrics import AdversarialMetrics  # Add this import
import logging


def parse_test_args():
    parser = argparse.ArgumentParser(
        description='Simplified Testing Configuration')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset name to test')
    parser.add_argument('--arch', type=str, required=True,
                        help='Model architecture to test')
    parser.add_argument('--depth', type=str, required=True,
                        help='JSON string for model depth')
    parser.add_argument('--model_path', type=str,
                        required=True, help='Path to saved model weights')
    # Optional argument for a single image path
    parser.add_argument('--image_path', type=str,
                        help='Optional path to a single image for prediction')
    # Set defaults for other parameters:
    parser.add_argument('--test_batch', type=int,
                        default=32, help='Test batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory')
    parser.add_argument('--gpu-ids', default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--device-index', type=int,
                        default=0, help='Primary GPU index to use')
    parser.add_argument('--task_name', type=str,
                        default='normal_training', help='Task name')
    # Add adversarial evaluation arguments
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable adversarial evaluation')
    parser.add_argument('--attack_type', type=str, default='fgsm',
                        choices=['fgsm', 'pgd', 'bim', 'jsma'],
                        help='Type of attack for evaluation')
    parser.add_argument('--attack_eps', type=float, default=0.3,
                        help='Epsilon for adversarial attacks')
    return parser.parse_args()


def main():
    args = parse_test_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Process the depth argument
    try:
        args.depth = json.loads(args.depth)
    except Exception as e:
        logging.error(f"Invalid depth JSON: {str(e)}")
        return

    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device_index}" if use_cuda else "cpu")
    args.device = device

    # Load test dataset using DatasetLoader (dummy loaders for train/val)
    dataset_loader = DatasetLoader()
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data,
        batch_size={'train': args.test_batch,
                    'val': args.test_batch, 'test': args.test_batch},
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Build model using ModelLoader via TrainingManager for consistency
    manager = TrainingManager(args)
    num_classes = len(test_loader.dataset.classes)
    models_and_names = manager.model_loader.get_model(
        model_name=args.arch,
        depth=args.depth,
        input_channels=3,
        num_classes=num_classes,
        task_name=args.task_name,
        dataset_name=args.data
    )
    if not models_and_names:
        logging.error("No model returned from ModelLoader")
        return
    model, model_name = models_and_names[0]

    # Initialize adversarial components if needed
    if args.adversarial:
        args.attack_name = args.attack_type
        args.epsilon = args.attack_eps
        adv_metrics = AdversarialMetrics()

    # Create a Trainer instance with dummy train/val loaders
    trainer = Trainer(
        model=model,
        train_loader=test_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        model_name=model_name,
        task_name=args.task_name,
        dataset_name=args.data,
        device=device,
        config=args,
        scheduler=None
    )
    trainer.load_model(args.model_path)

    # Run test on full dataset and log metrics
    if args.adversarial:
        test_loss, test_accuracy, adv_test_loss, adv_test_accuracy = trainer.test()
        logging.info(
            f"Test results for {model_name}:\n"
            f"Clean  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}\n"
            f"Advers - Loss: {adv_test_loss:.4f}, Accuracy: {adv_test_accuracy:.4f}"
        )
    else:
        test_loss, test_accuracy = trainer.test()
        logging.info(
            f"Test results for {model_name}: Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}"
        )

    # Retrieve test results for further metric evaluation and visualization
    true_labels, logits = trainer.get_test_results()
    if logits.ndim == 1:
        # Binary case: create discrete predictions and probability scores
        discrete_preds = (logits > 0.5).astype(int)
        # Also convert to 2-col probability format for ROC, etc.
        prob_preds = np.vstack([1 - logits, logits]).T
    elif logits.ndim == 2 and logits.shape[1] > 1:
        discrete_preds = np.argmax(logits, axis=1)
        prob_preds = F.softmax(torch.tensor(logits), dim=1).numpy()
    else:
        discrete_preds = logits
        prob_preds = logits

    # Prepare dictionaries for visualization and evaluation
    true_labels_dict = {model_name: true_labels}
    discrete_preds_dict = {model_name: discrete_preds}
    prob_preds_dict = {model_name: prob_preds}

    # Evaluate additional metrics using discrete predictions (and pass probabilities for ROC/AUC)
    evaluator = Evaluator(model_name, [], true_labels, discrete_preds,
                          args.task_name, all_probabilities=prob_preds)
    evaluator.evaluate(args.data)

    # Visualization: pass a tuple containing true labels, discrete predictions, and probability scores.
    # (Update your Visualization.visualize_normal method to unpack this tuple accordingly.)
    viz = Visualization()
    viz.visualize_normal(
        [model_name],
        (true_labels_dict, discrete_preds_dict, prob_preds_dict),
        args.task_name,
        args.data,
        test_loader.dataset.classes
    )

    # If adversarial evaluation is enabled, visualize adversarial metrics
    if args.adversarial:
        # Update metrics with both clean and adversarial results
        adv_metrics.update_adversarial_comparison(
            phase='test',
            clean_loss=test_loss,
            clean_acc=test_accuracy,
            adv_loss=adv_test_loss,
            adv_acc=adv_test_accuracy
        )

        # Visualize adversarial training curves
        viz.visualize_adversarial_training(
            adv_metrics.metrics,
            args.task_name,
            args.data,
            model_name
        )

    # If an image path is provided, perform single image inference
    if args.image_path:
        logging.info(
            f"Performing single image inference for {args.image_path}")
        try:
            image = Image.open(args.image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            return
        # Get the transform from the test dataset
        test_transform = test_loader.dataset.transform
        image_tensor = test_transform(image).unsqueeze(0).to(device)
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
        predicted_class = test_loader.dataset.classes[predicted_idx.item()]

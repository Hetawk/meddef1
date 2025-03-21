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

from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader
from utils.visual.visualization import Visualization
# Import the centralized argument parser
from argument_parser import parse_args

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

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

    return np.array(all_preds), np.array(all_targets)


def main():
    # Parse arguments with mode='test'
    args = parse_args(mode='test')

    # Load the dataset
    dataset_loader = DatasetLoader()
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data[0],
        batch_size={'test': args.batch_size},
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Get number of classes
    dataset = test_loader.dataset
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, 'class_to_idx'):
        num_classes = len(dataset.class_to_idx)
    else:
        raise AttributeError("Dataset does not contain class information")

    # Load the model
    model, model_name = load_model(args, num_classes)

    # Test the model
    all_preds, all_targets = test_model(model, test_loader, args)

    # Calculate metrics
    accuracy = (all_preds.flatten() == all_targets).mean()
    logger.info(f"Test accuracy: {accuracy:.4f}")

    # Generate classification report
    class_names = dataset.classes if hasattr(dataset, 'classes') else [
        str(i) for i in range(num_classes)]
    report = classification_report(
        all_targets, all_preds, target_names=class_names)
    logger.info("Classification Report:\n" + report)

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Save predictions if requested
    if args.save_predictions:
        pred_df = pd.DataFrame({
            'true': all_targets.flatten(),
            'pred': all_preds.flatten()
        })
        pred_df.to_csv(os.path.join(args.output_dir,
                       f"{model_name}_predictions.csv"), index=False)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"{model_name}_confusion_matrix.png"))

    # Additional analysis and visualizations could be added here

    logger.info(f"Testing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

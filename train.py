import argparse
import logging
import os
from datetime import datetime
import random
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.adv_metrics import AdversarialMetrics
from utils.training_logger import TrainingLogger
from utils.robustness.regularization import Regularization
from utils.timer import Timer
from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader
from utils.robustness.optimizers import OptimizerLoader
from utils.robustness.lr_scheduler import LRSchedulerLoader
from utils.utility import get_model_params
from utils.weighted_losses import WeightedCrossEntropyLoss, AggressiveMinorityWeightedLoss, DynamicSampleWeightedLoss
from utils.metrics import Metrics
from utils.evaluator import Evaluator
# Changed to import from argument_parser.py
from argument_parser import parse_args
import json
import warnings  # added import
from tqdm import tqdm   # Added import for progress bar
import torch.backends.cudnn  # Add this to ensure cudnn is recognized
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter

# Try importing TensorBoardLogger, fall back to MetricsLogger if that fails
try:
    from utils.tensorboard_wrapper import TensorBoardLogger
    USE_TENSORBOARD = True
    logging.info("Successfully imported TensorBoardLogger")
except ImportError:
    from utils.metrics_logger import MetricsLogger
    USE_TENSORBOARD = False
    logging.info("TensorBoard not available, using MetricsLogger instead")

# added to suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def _init_history():
    return {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'duration': [],
        'true_labels': [],
        'predictions': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_predictions': [],
        'val_targets': [],
        # New adversarial metrics
        'adv_loss': [],
        'adv_accuracy': [],
        'adv_predictions': [],
        'adv_targets': [],
        # New detailed metrics
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_metrics': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_per_class_metrics': []
    }


class Trainer:
    """Training orchestrator that handles the training loop and logging"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, model_name, task_name, dataset_name,
                 device, config, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.device = device
        self.config = config

        # Set up loss function based on config
        loss_type = getattr(config, 'loss_type', 'standard')
        if loss_type == 'standard':
            self.criterion = criterion
            logging.info("Using standard CrossEntropy loss")
        else:
            logging.info(f"Using {loss_type} loss function")

            # Get all targets from the dataset for class distribution analysis
            try:
                # Try to get targets from train_loader.dataset.targets (common for ImageFolder)
                if hasattr(train_loader.dataset, 'targets'):
                    dataset_targets = train_loader.dataset.targets
                # Try to get targets from TensorDataset
                elif hasattr(train_loader.dataset, 'tensors') and len(train_loader.dataset.tensors) > 1:
                    dataset_targets = train_loader.dataset.tensors[1]
                else:
                    # Collect targets by iterating through dataloader (slower but works)
                    dataset_targets = []
                    logging.info(
                        "Collecting class distribution from dataloader...")
                    for _, batch_targets in train_loader:
                        dataset_targets.extend(batch_targets.cpu().numpy())
                    dataset_targets = np.array(dataset_targets)

                # Initialize the appropriate loss function based on loss_type
                if loss_type == 'weighted':
                    self.criterion = WeightedCrossEntropyLoss(
                        dataset=dataset_targets)
                elif loss_type == 'aggressive':
                    self.criterion = AggressiveMinorityWeightedLoss(
                        dataset=dataset_targets)
                elif loss_type == 'dynamic':
                    alpha = getattr(config, 'focal_alpha', 0.5)
                    gamma = getattr(config, 'focal_gamma', 2.0)
                    self.criterion = DynamicSampleWeightedLoss(
                        max_epochs=getattr(config, 'epochs', 100),
                        alpha=alpha, gamma=gamma)

                logging.info(
                    f"Successfully initialized {loss_type} loss function")
            except Exception as e:
                logging.error(
                    f"Error initializing weighted loss: {e}. Falling back to standard loss.")
                self.criterion = criterion

        self.has_trained = False
        self.epochs = getattr(config, 'epochs', 100)
        self.lambda_l2 = getattr(config, 'lambda_l2', 1e-4)
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self.args = config
        if not hasattr(self.args, 'lr'):
            self.args.lr = 0.001

        self.scaler = GradScaler()
        self.timer = Timer()
        self.training_logger = TrainingLogger()
        self.history = _init_history()

        from utils.visual.visualization import Visualization
        self.visualization = Visualization()

        self.model.to(self.device)
        if hasattr(config, 'drop'):
            Regularization.apply_dropout(self.model, config.drop)

        self.adversarial = getattr(config, 'adversarial', False)
        if self.adversarial:
            from gan.defense.adv_train import AdversarialTraining
            if not hasattr(config, 'attack_name'):
                config.attack_name = getattr(config, 'attack_type', 'fgsm')
            if not hasattr(config, 'epsilon'):
                config.epsilon = getattr(config, 'attack_eps', 0.3)
            self.adversarial_trainer = AdversarialTraining(
                model, criterion, config)
            logging.info(
                f"Training {self.model_name} with adversarial training...")

        # Initialize tracking variables for metrics
        self.error_if_nonfinite = False
        self.val_loss = float('inf')
        self.current_lr = self.args.lr
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0  # Initialize best F1 score for tracking
        self.best_val_balanced_acc = 0.0  # Initialize best balanced accuracy
        self.no_improvement_count = 0
        self.adv_metrics = AdversarialMetrics()

        # Initialize the lists here for test results
        self.true_labels = []
        self.predictions = []
        self.adv_predictions = []

        # Use either TensorBoardLogger or MetricsLogger depending on availability
        if USE_TENSORBOARD:
            self.tb_logger = TensorBoardLogger(
                task_name, dataset_name, model_name)
        else:
            self.tb_logger = MetricsLogger(task_name, dataset_name, model_name)

        # Log hyperparameters to TensorBoard
        hparams = {
            'learning_rate': getattr(config, 'lr', 0.001),
            'batch_size': getattr(config, 'train_batch', 32),
            'optimizer': getattr(config, 'optimizer', 'adam'),
            'model': model_name,
            'loss_type': getattr(config, 'loss_type', 'standard'),
            'weight_decay': getattr(config, 'weight_decay', 1e-4),
            'dropout': getattr(config, 'drop', 0.0),
            'scheduler': getattr(config, 'scheduler', 'none')
        }
        self.tb_logger.log_hparams(hparams)

        # Add new attributes for balanced metrics and class-specific metrics
        self.class_names = self._get_class_names(train_loader)
        self.per_class_metrics = getattr(config, 'per_class_metrics', True)

        # Create a threshold tracker for binary classification
        self.threshold = 0.5
        self.optimize_threshold = getattr(config, 'optimize_threshold', False)

        # Add history tracking for detailed metrics
        self.history['precision'] = []
        self.history['recall'] = []
        self.history['f1'] = []
        self.history['per_class_metrics'] = []
        self.history['val_precision'] = []
        self.history['val_recall'] = []
        self.history['val_f1'] = []
        self.history['val_per_class_metrics'] = []

        # Create sampler attribute for potential reweighting
        self.use_weighted_sampler = getattr(
            config, 'use_weighted_sampler', False)
        self.sampler = None

        # Use weighted sampler if specified
        if self.use_weighted_sampler:
            self._setup_weighted_sampler(train_loader)

    def _get_class_names(self, data_loader):
        """Get class names from the dataset if available"""
        if hasattr(data_loader.dataset, 'classes'):
            return data_loader.dataset.classes
        elif hasattr(data_loader.dataset, 'class_to_idx'):
            # Map indices back to class names
            class_to_idx = data_loader.dataset.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return [idx_to_class.get(i, str(i)) for i in range(len(class_to_idx))]
        else:
            # Fallback to generic class names
            # Try to infer number of classes from the criterion
            if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
                num_classes = len(self.criterion.weight)
            else:
                # Try to infer from the model's final layer
                if hasattr(self.model, 'fc'):
                    num_classes = self.model.fc.out_features
                elif hasattr(self.model, 'head'):
                    num_classes = self.model.head.out_features
                else:
                    num_classes = 2  # Default to binary classification
            return [f"Class {i}" for i in range(num_classes)]

    def _setup_weighted_sampler(self, train_loader):
        """Setup WeightedRandomSampler based on class distribution"""
        logging.info("Setting up WeightedRandomSampler for imbalanced data...")

        try:
            # Extract targets/labels from the dataset
            targets = []
            if hasattr(train_loader.dataset, 'targets'):
                targets = train_loader.dataset.targets
            elif hasattr(train_loader.dataset, 'tensors') and len(train_loader.dataset.tensors) > 1:
                targets = train_loader.dataset.tensors[1].tolist()
            else:
                # Extract targets by iterating through the dataset
                for _, target in train_loader.dataset:
                    if torch.is_tensor(target):
                        targets.append(target.item())
                    else:
                        targets.append(target)

            # Count class frequencies
            class_counts = Counter(targets)
            logging.info(f"Class distribution: {dict(class_counts)}")

            # Calculate class weights (inverse frequency)
            total_samples = len(targets)
            class_weights = {class_idx: total_samples /
                             count for class_idx, count in class_counts.items()}

            # Assign weights to each sample
            weights = [class_weights[target] for target in targets]
            weights = torch.DoubleTensor(weights)

            # Create the sampler
            self.sampler = WeightedRandomSampler(
                weights, len(weights), replacement=True)

            # Create new train loader with the sampler
            self.train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=self.sampler,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory
            )

            logging.info("WeightedRandomSampler successfully created")
        except Exception as e:
            logging.error(f"Failed to create WeightedRandomSampler: {e}")

    def calculate_detailed_metrics(self, true_labels, predictions, probabilities=None, phase="train"):
        """Calculate detailed metrics including per-class performance"""
        detailed_metrics = {}

        # Convert tensors to numpy arrays if needed
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        # Calculate overall metrics
        metrics_dict = Metrics.calculate_metrics(
            true_labels, predictions, probabilities)
        detailed_metrics.update(metrics_dict)

        # Add per-class metrics if requested
        if self.per_class_metrics:
            per_class = {}
            classes = np.unique(true_labels)

            for cls in classes:
                cls_mask = (true_labels == cls)
                if sum(cls_mask) == 0:
                    continue

                # Calculate binary metrics for this class (one-vs-rest)
                cls_true = (true_labels == cls).astype(int)
                cls_pred = (predictions == cls).astype(int)

                # Basic metrics per class
                try:
                    cls_metrics = {
                        'accuracy': Metrics.to_numpy(np.mean(cls_true == cls_pred)),
                        'precision': Metrics.calculate_metrics(cls_true, cls_pred)['precision'],
                        'recall': Metrics.calculate_metrics(cls_true, cls_pred)['recall'],
                        'f1': Metrics.calculate_metrics(cls_true, cls_pred)['f1'],
                        'support': np.sum(cls_true)
                    }

                    class_name = self.class_names[cls] if cls < len(
                        self.class_names) else f"Class {cls}"
                    per_class[class_name] = cls_metrics
                except Exception as e:
                    logging.warning(
                        f"Error calculating metrics for class {cls}: {e}")

            detailed_metrics['per_class'] = per_class

        # If it's binary classification and we have probabilities, optimize threshold
        if len(np.unique(true_labels)) == 2 and probabilities is not None and self.optimize_threshold:
            if phase == "val":
                self._optimize_threshold(true_labels, probabilities)

            # Apply optimal threshold to predictions
            if probabilities.ndim > 1:
                # Use probability of positive class
                opt_preds = (probabilities[:, 1] > self.threshold).astype(int)
            else:
                opt_preds = (probabilities > self.threshold).astype(int)

            # Calculate metrics with optimized threshold
            opt_metrics = Metrics.calculate_metrics(true_labels, opt_preds)
            detailed_metrics['optimized_threshold'] = self.threshold
            detailed_metrics['optimized_accuracy'] = opt_metrics['accuracy']
            detailed_metrics['optimized_f1'] = opt_metrics['f1']

        return detailed_metrics

    def _optimize_threshold(self, true_labels, probabilities):
        """Find optimal threshold for binary classification"""
        # Handle both one-hot and regular probabilities
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            probs = probabilities[:, 1]  # Probability of positive class
        else:
            probs = probabilities

        # Get precision, recall, thresholds
        precision, recall, thresholds = precision_recall_curve(
            true_labels, probs)

        # Calculate F1 score for each threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            self.threshold = thresholds[best_idx]
        else:
            self.threshold = 0.5  # Default if something went wrong

        logging.info(
            f"Optimized threshold: {self.threshold:.4f} with F1: {f1_scores[best_idx]:.4f}")

    def _visualize_metrics(self, metrics, epoch, phase="train"):
        """Create and save visualizations for metrics with proper resource handling"""
        if not hasattr(self, 'visualization'):
            return

        # Only create visualizations periodically to save resources
        if epoch % 10 != 0 and epoch != self.epochs - 1:
            return

        # Create directory for visualizations
        viz_dir = os.path.join('out', self.task_name, self.dataset_name,
                               self.model_name, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        try:
            # Create confusion matrix plot
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                plt.colorbar()

                # Add class labels
                classes = self.class_names if len(self.class_names) == len(cm) else [
                    f"Class {i}" for i in range(len(cm))]
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)

                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(os.path.join(
                    viz_dir, f'confusion_matrix_{phase}_epoch_{epoch+1}.png'))
                plt.close(fig)  # Explicitly close figure

            # Create per-class metrics bar chart
            if 'per_class' in metrics:
                per_class = metrics['per_class']
                metrics_to_plot = ['precision', 'recall', 'f1']

                fig = plt.figure(figsize=(12, 6))
                x = np.arange(len(per_class))
                width = 0.25

                for i, metric in enumerate(metrics_to_plot):
                    values = [per_class[cls][metric] for cls in per_class]
                    plt.bar(x + width * (i - 1), values,
                            width, label=metric.capitalize())

                plt.xlabel('Class')
                plt.ylabel('Score')
                plt.title(f'Per-Class Metrics - Epoch {epoch+1}')
                plt.xticks(x, per_class.keys(), rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(
                    viz_dir, f'per_class_metrics_{phase}_epoch_{epoch+1}.png'))
                plt.close(fig)  # Explicitly close figure

        except Exception as e:
            logging.warning(f"Failed to create visualization: {e}")
            plt.close('all')  # Close any open figures on error

    def train(self, patience):
        if self.has_trained:
            logging.warning(
                f"{self.model} has already been trained. Training again will overwrite the existing model.")
            return
        logging.info(f"Training {self.model_name}...")
        self.has_trained = True

        torch.cuda.empty_cache()
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(self.device)

        self.model.train()
        total_batches = len(self.train_loader)
        initial_params = get_model_params(self.model)
        logging.info(f"Initial model parameters: {initial_params:.2f}M")

        # Configure early stopping
        min_epochs = getattr(self.args, 'min_epochs', 0)
        early_stopping_metric = getattr(
            self.args, 'early_stopping_metric', 'f1')  # Default to f1
        saved_attacks = False

        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # Log model graph to TensorBoard (try with a sample batch)
        try:
            sample_input = next(iter(self.train_loader))[0][:1].to(self.device)
            self.tb_logger.log_model_graph(self.model, sample_input)
        except Exception as e:
            logging.warning(
                f"Could not add model graph to TensorBoard: {str(e)}")

        # Add tracking for detailed metrics
        for epoch in range(self.epochs):
            # Update epoch for dynamic sample weighting if needed
            if isinstance(self.criterion, DynamicSampleWeightedLoss):
                self.criterion.update_epoch(epoch)
                logging.info(
                    f"Updated dynamic loss function for epoch {epoch+1}")

            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            batch_loss = 0.0
            adv_loss_sum = 0.0
            adv_correct = 0  # Initialize adv_correct
            # Prepare lists to log epoch results
            epoch_true_labels = []
            epoch_predictions = []
            epoch_probabilities = []
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")):
                try:
                    # Data preparation: remove unused batch_indices
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device, non_blocking=True)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.device, non_blocking=True)

                    if self.adversarial and not saved_attacks and epoch == 0 and batch_idx == 0:
                        orig, adv_data, _ = self.adversarial_trainer.attack.attack(
                            data, target)
                        self.adversarial_trainer.save_attack_samples(
                            orig, adv_data)
                        saved_attacks = True

                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        if self.adversarial:
                            if hasattr(self.adversarial_trainer.attack, 'generate'):
                                adv_data = self.adversarial_trainer.attack.generate(
                                    data, target, self.args.epsilon)
                            else:
                                _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                    data, target)
                            with autocast():
                                adv_batch_loss = self.criterion(
                                    self.model(adv_data), target)
                            adv_loss_sum += adv_batch_loss.item()
                            adv_pred = self.model(adv_data).argmax(
                                dim=1, keepdim=True)
                            adv_correct += adv_pred.eq(
                                target.view_as(adv_pred)).sum().item()
                            w = float(self.args.adv_weight)
                            loss = (1 - w) * loss + w * adv_batch_loss
                        loss = loss / self.accumulation_steps
                    if not torch.isfinite(loss):
                        logging.debug(
                            f"Non-finite loss encountered at batch {batch_idx}. Skipping batch.")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    self.scaler.scale(loss).backward()
                    with torch.no_grad():
                        batch_loss += loss.item() * self.accumulation_steps
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        epoch_true_labels.extend(target.cpu().numpy())
                        epoch_predictions.extend(pred.cpu().numpy())
                        output_probs = torch.nn.functional.softmax(
                            output, dim=1)
                        epoch_probabilities.extend(output_probs.cpu().numpy())
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(
                        ), max_norm=self.args.max_grad_norm, error_if_nonfinite=False)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        epoch_loss += batch_loss
                        batch_loss = 0.0
                    if batch_idx % 100 == 0:
                        logging.info(
                            f'Epoch: {epoch+1}/{self.epochs} | Batch: {batch_idx * len(data)}/{len(self.train_loader.dataset)} | Loss: {loss.item():.4f} | Accuracy: {correct/total if total else 0:.4f}')
                except RuntimeError as err:
                    logging.error(f"Runtime error in batch {batch_idx}: {err}")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler = GradScaler()
                    continue
                except Exception as exp:
                    logging.exception(
                        f"Unexpected error in batch {batch_idx}: {exp}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

            val_loss, val_accuracy = self.validate()
            # Run validation with detailed metrics to get F1 and other scores
            val_loss, val_accuracy, val_detailed_metrics = self.validate_with_metrics()

            # Extract F1 score and balanced accuracy from detailed metrics
            val_f1 = val_detailed_metrics.get('f1', 0.0)
            val_balanced_acc = val_detailed_metrics.get(
                'balanced_accuracy', 0.0)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use appropriate metric for scheduler
                    if early_stopping_metric == 'f1':
                        # Negate because ReduceLROnPlateau minimizes
                        self.scheduler.step(-val_f1)
                    elif early_stopping_metric == 'balanced_acc':
                        # Negate because ReduceLROnPlateau minimizes
                        self.scheduler.step(-val_balanced_acc)
                    else:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model based on validation performance using the selected metric
            improved = False

            if early_stopping_metric == 'loss':
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_accuracy
                    self.best_val_f1 = val_f1
                    self.best_val_balanced_acc = val_balanced_acc
                    improved = True
            elif early_stopping_metric == 'f1':
                if val_f1 > self.best_val_f1:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_accuracy
                    self.best_val_f1 = val_f1
                    self.best_val_balanced_acc = val_balanced_acc
                    improved = True
            elif early_stopping_metric == 'balanced_acc':
                if val_balanced_acc > self.best_val_balanced_acc:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_accuracy
                    self.best_val_f1 = val_f1
                    self.best_val_balanced_acc = val_balanced_acc
                    improved = True
            else:  # Default to accuracy
                if val_accuracy > self.best_val_acc:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_accuracy
                    self.best_val_f1 = val_f1
                    self.best_val_balanced_acc = val_balanced_acc
                    improved = True

            if improved:
                self.no_improvement_count = 0
                self.save_model(
                    f"save_model/best_{self.model_name}_{self.dataset_name}.pth")
                logging.info(
                    f"Improved {early_stopping_metric}! Saving model.")
            else:
                self.no_improvement_count += 1
                logging.info(
                    f"No improvement in {early_stopping_metric} for {self.no_improvement_count} epochs.")

            # Only apply early stopping after minimum epochs
            if epoch >= min_epochs and self.no_improvement_count >= patience:
                logging.info(
                    f"Early stopping triggered after {epoch + 1} epochs (minimum epochs: {min_epochs})")
                break

            epoch_acc = correct / total if total > 0 else 0
            adv_accuracy = (
                adv_correct / total) if (self.adversarial and total > 0) else 0
            avg_adv_loss = (adv_loss_sum / len(self.train_loader)
                            ) if self.adversarial else 0
            self.adv_metrics.update_adversarial_comparison(phase='train', clean_loss=epoch_loss / len(
                self.train_loader), clean_acc=epoch_acc, adv_loss=avg_adv_loss, adv_acc=adv_accuracy)
            if self.adversarial:
                logging.info(
                    f'Epoch {epoch+1} Training - Clean: Loss={epoch_loss/len(self.train_loader):.4f}, Acc={epoch_acc:.4f} | Adversarial: Loss={avg_adv_loss:.4f}, Acc={adv_accuracy:.4f}')
            else:
                logging.info(
                    f'Epoch {epoch+1} Training - Loss={epoch_loss/len(self.train_loader):.4f}, Acc={epoch_acc:.4f}')
            self._update_history(epoch, epoch_loss, correct, total, val_loss, val_accuracy,
                                 epoch_true_labels, epoch_predictions, 0)  # duration not used
            self.visualization.visualize_adversarial_training(
                self.adv_metrics.metrics, self.task_name, self.dataset_name, self.model_name)

            # Log epoch metrics to TensorBoard
            epoch_acc = correct / total if total > 0 else 0
            epoch_loss = epoch_loss / len(self.train_loader)
            self.tb_logger.log_epoch_metrics(
                epoch, epoch_loss, epoch_acc,
                val_loss, val_accuracy,
                self.current_lr,
                avg_adv_loss if self.adversarial else None,
                adv_accuracy if self.adversarial else None
            )

            # Calculate and log detailed metrics
            detailed_metrics = self.calculate_detailed_metrics(
                np.array(epoch_true_labels),
                np.array(epoch_predictions),
                np.array(epoch_probabilities),
                phase="train"
            )

            # Update history with detailed metrics
            self.history['precision'].append(
                detailed_metrics.get('precision', 0))
            self.history['recall'].append(detailed_metrics.get('recall', 0))
            self.history['f1'].append(detailed_metrics.get('f1', 0))
            self.history['per_class_metrics'].append(
                detailed_metrics.get('per_class', {}))

            # Visualize metrics periodically
            self._visualize_metrics(detailed_metrics, epoch, phase="train")

            # Log detailed training metrics
            logging.info(f"Epoch {epoch+1} Training - Accuracy: {detailed_metrics['accuracy']:.4f}, "
                         f"Precision: {detailed_metrics['precision']:.4f}, "
                         f"Recall: {detailed_metrics['recall']:.4f}, "
                         f"F1: {detailed_metrics['f1']:.4f}")

            # Log per-class metrics if available
            if 'per_class' in detailed_metrics:
                per_class = detailed_metrics['per_class']
                for cls_name, cls_metrics in per_class.items():
                    logging.info(f"  {cls_name}: F1={cls_metrics['f1']:.4f}, "
                                 f"Precision={cls_metrics['precision']:.4f}, "
                                 f"Recall={cls_metrics['recall']:.4f}, "
                                 f"Support={cls_metrics['support']}")

            # Run validation with detailed metrics
            val_loss, val_accuracy, val_detailed_metrics = self.validate_with_metrics()

            # Update history with validation metrics
            self.history['val_precision'].append(
                val_detailed_metrics.get('precision', 0))
            self.history['val_recall'].append(
                val_detailed_metrics.get('recall', 0))
            self.history['val_f1'].append(val_detailed_metrics.get('f1', 0))
            self.history['val_per_class_metrics'].append(
                val_detailed_metrics.get('per_class', {}))

            # Log all key metrics clearly
            logging.info(f"Epoch {epoch+1} Validation - "
                         f"Loss: {val_loss:.4f}, "
                         f"Accuracy: {val_accuracy:.4f}, "
                         f"F1: {val_f1:.4f}, "
                         f"Balanced Acc: {val_balanced_acc:.4f}")

            # ...existing code...

        # After training complete, add final test results
        test_results = self.test()
        if self.adversarial:
            test_loss, test_accuracy, adv_test_loss, adv_test_accuracy = test_results
            self.tb_logger.log_test_results(
                test_loss, test_accuracy, adv_test_loss, adv_test_accuracy)
        else:
            test_loss, test_accuracy = test_results
            self.tb_logger.log_test_results(test_loss, test_accuracy)

        # After training complete, log best metrics
        logging.info(f"Training finished. Best metrics - "
                     f"Loss: {self.best_val_loss:.4f}, "
                     f"Accuracy: {self.best_val_acc:.4f}, "
                     f"F1: {self.best_val_f1:.4f}, "
                     f"Balanced Acc: {self.best_val_balanced_acc:.4f}")

        # Close the TensorBoard logger
        self.tb_logger.close()

        return self.best_val_loss, self.best_val_acc, self.best_val_f1

    def _log_training_progress(self, epoch, batch_idx, data, loss, correct, total, start_time):
        accuracy = correct / total if total > 0 else 0
        current_time = datetime.now()
        duration = Timer.format_duration(
            (current_time - start_time).total_seconds())
        logging.info(
            f'Epoch: {epoch+1}/{self.epochs} | Batch: {batch_idx * len(data)}/{len(self.train_loader.dataset)} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f} | Duration: {duration}')

    def _update_history(self, epoch, epoch_loss, correct, total, val_loss, val_accuracy, epoch_true_labels, epoch_predictions, start_time):
        accuracy = correct / total if total > 0 else 0
        end_time = datetime.now()
        epoch_duration = Timer.format_duration(
            (end_time - start_time).total_seconds()) if start_time else None
        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(epoch_loss)
        self.history['accuracy'].append(accuracy)
        self.history['duration'].append(epoch_duration)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        self.history['true_labels'].append(epoch_true_labels)
        self.history['predictions'].append(epoch_predictions)
        self.history['val_predictions'].append([])  # placeholder
        self.history['val_targets'].append([])        # placeholder

    def validate(self):
        self.model.eval()
        val_loss = 0
        adv_val_loss = 0
        correct = 0
        adv_correct = 0
        total = 0
        val_predictions = []
        val_targets = []
        adv_accuracy = 0  # Initialize
        try:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.val_loader):
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device, non_blocking=True)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.device, non_blocking=True)
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    val_loss += loss.item()
                    if self.adversarial:
                        with torch.enable_grad():
                            if hasattr(self.adversarial_trainer.attack, 'generate'):
                                adv_data = self.adversarial_trainer.attack.generate(
                                    data, target, self.args.epsilon)
                            else:
                                _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                    data, target)
                        with autocast():
                            adv_output = self.model(adv_data)
                            adv_loss = self.criterion(adv_output, target)
                        adv_val_loss += adv_loss.item()
                        adv_pred = adv_output.argmax(dim=1, keepdim=True)
                        adv_correct += adv_pred.eq(
                            target.view_as(adv_pred)).sum().item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    val_predictions.extend(pred.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
                    if batch_idx % 100 == 0:
                        logging.debug(
                            f'Validation Batch: {batch_idx}/{len(self.val_loader)}')
            val_loss /= len(self.val_loader)
            accuracy = correct / total if total > 0 else 0
            if self.adversarial:
                adv_val_loss /= len(self.val_loader)
                adv_accuracy = adv_correct / total if total > 0 else 0
                logging.info(
                    f'Validation - Clean: Loss={val_loss:.4f}, Acc={accuracy:.4f} | Adversarial: Loss={adv_val_loss:.4f}, Acc={adv_accuracy:.4f}')
            else:
                logging.info(
                    f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
            self.history['val_predictions'].append(val_predictions)
            self.history['val_targets'].append(val_targets)
            self.adv_metrics.update_adversarial_comparison(phase='val', clean_loss=val_loss, clean_acc=accuracy,
                                                           adv_loss=adv_val_loss if self.adversarial else 0, adv_acc=adv_accuracy if self.adversarial else 0)
        except Exception as e:
            logging.error(f"Error during validation: {e}")
            return float('inf'), 0.0
        finally:
            self.model.train()
        return val_loss, accuracy

    def validate_with_metrics(self):
        """Validate with detailed metrics calculation"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_predictions = []
        val_targets = []
        val_probabilities = []

        try:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.val_loader):
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device, non_blocking=True)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.device, non_blocking=True)

                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)

                    val_loss += loss.item()

                    # Get probabilities and predictions
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred = output.argmax(dim=1, keepdim=True)

                    # Update metrics
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                    # Store results for detailed metrics
                    val_predictions.extend(pred.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
                    val_probabilities.extend(probs.cpu().numpy())

            val_loss /= len(self.val_loader)
            accuracy = correct / total if total > 0 else 0

            # Calculate detailed metrics
            detailed_metrics = self.calculate_detailed_metrics(
                np.array(val_targets),
                np.array(val_predictions),
                np.array(val_probabilities),
                phase="val"
            )

            # Visualize metrics
            self._visualize_metrics(detailed_metrics, epoch=0, phase="val")

            # Log detailed metrics
            logging.info(f"Validation - Loss: {val_loss:.4f}, "
                         f"Accuracy: {accuracy:.4f}, "
                         f"Precision: {detailed_metrics['precision']:.4f}, "
                         f"Recall: {detailed_metrics['recall']:.4f}, "
                         f"F1: {detailed_metrics['f1']:.4f}")

            # Log per-class metrics
            if 'per_class' in detailed_metrics:
                per_class = detailed_metrics['per_class']
                for cls_name, cls_metrics in per_class.items():
                    logging.info(f"  {cls_name}: F1={cls_metrics['f1']:.4f}, "
                                 f"Precision={cls_metrics['precision']:.4f}, "
                                 f"Recall={cls_metrics['recall']:.4f}")

            return val_loss, accuracy, detailed_metrics

        except Exception as e:
            logging.error(f"Error during validation with metrics: {e}")
            return float('inf'), 0.0, {}
        finally:
            self.model.train()

    def test(self):
        """Enhanced testing with detailed metrics"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # Store detailed results for metrics calculation
        self.true_labels = []
        self.predictions = []
        self.probabilities = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing", unit="batch"):
                if isinstance(data, torch.Tensor):
                    data = data.to(self.device)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device)

                # Forward pass
                output = self.model(data)
                test_loss += self.criterion(output, target).item()

                # Get probabilities and predictions
                probs = torch.nn.functional.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                # Update metrics
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                # Store for detailed metrics
                self.true_labels.extend(target.cpu().numpy())
                self.predictions.extend(pred.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())

                # ... existing adversarial code ...

        test_loss /= len(self.test_loader)
        accuracy = correct / total if total > 0 else 0

        # Calculate and log detailed metrics
        detailed_metrics = self.calculate_detailed_metrics(
            np.array(self.true_labels),
            np.array(self.predictions),
            np.array(self.probabilities),
            phase="test"
        )

        # Create evaluator and save metrics
        evaluator = Evaluator(
            model_name=self.model_name,
            results=[],
            true_labels=np.array(self.true_labels),
            all_predictions=np.array(self.predictions),
            task_name=self.task_name,
            all_probabilities=np.array(self.probabilities)
        )
        evaluator.save_metrics(detailed_metrics, self.dataset_name)

        # Create threshold optimization curve for binary classification
        if len(np.unique(self.true_labels)) == 2 and len(self.probabilities) > 0:
            self._create_threshold_curve(
                np.array(self.true_labels), np.array(self.probabilities))

        # Log detailed test metrics
        logging.info(f"Test Results - Loss: {test_loss:.4f}, "
                     f"Accuracy: {accuracy:.4f}, "
                     f"Precision: {detailed_metrics['precision']:.4f}, "
                     f"Recall: {detailed_metrics['recall']:.4f}, "
                     f"F1: {detailed_metrics['f1']:.4f}")

        # Log per-class metrics
        if 'per_class' in detailed_metrics:
            per_class = detailed_metrics['per_class']
            for cls_name, cls_metrics in per_class.items():
                logging.info(f"  {cls_name}: F1={cls_metrics['f1']:.4f}, "
                             f"Precision={cls_metrics['precision']:.4f}, "
                             f"Recall={cls_metrics['recall']:.4f}, "
                             f"Support={cls_metrics['support']}")

        self.model.train()

        # Include detailed metrics in the return value
        return (test_loss, accuracy, detailed_metrics) if not self.adversarial else (test_loss, accuracy, detailed_metrics, self.adv_test_loss, self.adv_test_accuracy)

    def _create_threshold_curve(self, true_labels, probabilities):
        """Create and save threshold optimization curve for binary classification"""
        try:
            # Get probabilities for positive class
            if probabilities.ndim > 1 and probabilities.shape[1] > 1:
                pos_probs = probabilities[:, 1]
            else:
                pos_probs = probabilities

            # Calculate precision, recall, thresholds
            precision, recall, thresholds = precision_recall_curve(
                true_labels, pos_probs)

            # Calculate F1 score for each threshold
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

            # Create directory for visualizations
            viz_dir = os.path.join('out', self.task_name, self.dataset_name,
                                   self.model_name, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)

            # Create precision-recall vs threshold curve
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1 Score')

            # Mark optimal threshold
            best_idx = np.argmax(f1_scores[:-1])
            best_threshold = thresholds[best_idx]
            plt.axvline(x=best_threshold, color='k', linestyle='-', alpha=0.3)
            plt.text(best_threshold, 0.5, f'Best threshold: {best_threshold:.4f}',
                     rotation=90, verticalalignment='center')

            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1 Score vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, 'threshold_optimization.png'))
            plt.close()

            # Create ROC curve
            fpr, tpr, _ = roc_curve(true_labels, pos_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, 'roc_curve.png'))
            plt.close()

            logging.info(f"Threshold optimization curves saved to {viz_dir}")
        except Exception as e:
            logging.error(f"Error creating threshold curve: {e}")

    def save_model(self, path):
        filename, ext = os.path.splitext(path)
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{filename}_epochs{self.epochs}_lr{self.args.lr}_batch{self.args.train_batch}_{timestamp}{ext}"
        if self.adversarial:
            path = os.path.join(
                'out', self.task_name, self.dataset_name, self.model_name, 'adv', filename)
        else:
            path = os.path.join('out', self.task_name,
                                self.dataset_name, self.model_name, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f'Model saved to {path}')

    def save_history_to_csv(self, filename):
        filename = os.path.join('out', self.task_name,
                                self.dataset_name, self.model_name, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        keys_to_check = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'duration',
                         'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
                         'true_labels', 'predictions']

        # Add check for empty history
        if not self.history['epoch']:
            logging.warning("No training history to save.")
            return

        # Verify all lists have the same length
        epoch_len = len(self.history['epoch'])
        for key in keys_to_check:
            if len(self.history[key]) != epoch_len:
                raise ValueError(
                    f"Length of {key} ({len(self.history[key])}) does not match length of 'epoch' ({epoch_len})")

        if len(self.history['true_labels']) != len(self.history['predictions']):
            raise ValueError(
                f"Length of true_labels ({len(self.history['true_labels'])}) does not match length of predictions ({len(self.history['predictions'])}).")

        self.history['model_name'] = [
            self.model_name] * len(self.history['epoch'])

        history_df = pd.DataFrame(self.history)
        history_df['true_labels'] = history_df['true_labels'].apply(
            lambda x: ','.join(map(str, x)))
        history_df['predictions'] = history_df['predictions'].apply(
            lambda x: ','.join(map(str, x)))

        if not os.path.isfile(filename):
            history_df.to_csv(filename, index=False)
        else:
            history_df.to_csv(filename, mode='a', index=False, header=False)

        logging.info(f'Training history saved to {filename}')

    def get_test_results(self):
        return np.array(self.true_labels), np.array(self.predictions)

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        new_state = {}
        for key, value in state.items():
            if key.endswith('weight_orig'):
                new_key = key[:-len('_orig')]
                new_state[new_key] = value
            elif key.endswith('weight_mask'):
                continue
            else:
                new_state[key] = value
        self.model.load_state_dict(new_state)
        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Loaded model from {path}")

    # Add a method to close the TensorBoard writer in case training is interrupted
    def __del__(self):
        """Clean up resources when the trainer is destroyed"""
        if hasattr(self, 'tb_logger'):
            self.tb_logger.close()

        # Close all matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')

        # Clean up visualization resources
        if hasattr(self, 'visualization') and hasattr(self.visualization, 'close_figures'):
            self.visualization.close_figures()


class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Setup random seed - simplified version
        seed = getattr(args, 'manualSeed', None)
        if seed is None:
            seed = random.randint(1, 10000)
        # Now this method accepts seed correctly
        self._setup_random_seeds(seed)

        # Initialize components
        self.model_loader = ModelLoader(
            args.device, args.arch,
            getattr(args, 'pretrained', True),  # Add default for pretrained
            getattr(args, 'fp16', False)        # Add default for fp16
        )
        self.dataset_loader = DatasetLoader()
        self.optimizer_loader = OptimizerLoader()
        self.lr_scheduler_loader = LRSchedulerLoader()

    # Remove @staticmethod so that 'seed' is passed in properly.
    def _setup_random_seeds(self, seed):
        """Setup random seeds for reproducibility"""
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_dataset(self, dataset_name):
        """Handle training for a specific dataset"""
        # Load dataset and get number of classes
        train_loader, val_loader, test_loader = self.dataset_loader.load_data(
            dataset_name=dataset_name,
            batch_size={
                'train': self.args.train_batch,
                'val': getattr(self.args, 'val_batch', self.args.train_batch),
                'test': getattr(self.args, 'test_batch', self.args.train_batch)
            },
            num_workers=self.args.num_workers,
            pin_memory=getattr(self.args, 'pin_memory', True)
        )

        # Get number of classes from the dataset: ensure that dataset has a 'classes' attribute.
        dataset = train_loader.dataset
        if hasattr(dataset, 'classes'):
            num_classes = len(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            num_classes = len(dataset.class_to_idx)
        else:
            raise AttributeError("Dataset does not contain class information.")

        # Log class distribution if using weighted loss
        if hasattr(self.args, 'loss_type') and self.args.loss_type != 'standard':
            try:
                class_counts = {}
                if hasattr(train_loader.dataset, 'targets'):
                    for target in train_loader.dataset.targets:
                        class_counts[target] = class_counts.get(target, 0) + 1
                elif hasattr(train_loader.dataset, 'samples'):
                    for _, target in train_loader.dataset.samples:
                        class_counts[target] = class_counts.get(target, 0) + 1

                if class_counts:
                    logging.info(
                        f"Class distribution for {dataset_name}: {class_counts}")
                    logging.info(
                        f"Using {self.args.loss_type} loss function for imbalanced data")
            except Exception as e:
                logging.warning(f"Couldn't analyze class distribution: {e}")

        # Get model for each architecture specified
        for arch in self.args.arch:
            try:
                # Now models_and_names is a list of (model, name) tuples
                models_and_names = self.model_loader.get_model(
                    model_name=arch,
                    depth=self.args.depth,
                    input_channels=3,
                    num_classes=num_classes,
                    task_name=self.args.task_name,
                    dataset_name=dataset_name
                )

                # Train each model variation
                for model, model_name in models_and_names:
                    # Create optimizer once and use it for both trainer and scheduler
                    optimizer = self.optimizer_loader.get_optimizer(
                        model, self.args)
                    scheduler = self.lr_scheduler_loader.get_scheduler(
                        optimizer, args=self.args)

                    # Create base criterion - may be replaced by weighted criterion in Trainer
                    base_criterion = torch.nn.CrossEntropyLoss()

                    trainer = Trainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        criterion=base_criterion,
                        model_name=model_name,
                        task_name=self.args.task_name,
                        dataset_name=dataset_name,
                        device=self.device,
                        config=self.args,
                        scheduler=scheduler
                    )

                    trainer.train(patience=self.args.patience)

                    # Handle both normal and adversarial test results
                    if self.args.adversarial:
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

            except Exception as e:
                logging.error(
                    f"Error training {arch} on {dataset_name}: {str(e)}")
                continue

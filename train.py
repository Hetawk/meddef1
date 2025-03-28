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
import torch.nn as nn
from utils.utility import get_model_params
from utils.weighted_losses import WeightedCrossEntropyLoss, AggressiveMinorityWeightedLoss, DynamicSampleWeightedLoss
from utils.metrics import Metrics
from utils.evaluator import Evaluator
from argument_parser import parse_args
import json
import warnings  # added import
from tqdm import tqdm   # Added import for progress bar
import torch.backends.cudnn  # Add this to ensure cudnn is recognized
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
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
        """Calculate metrics with reduced detail during training - ensuring basic metrics are always available"""
        # Import necessary metrics functions at the top of the function to ensure availability
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # Convert tensors to numpy arrays if needed
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        # During training, just calculate basic overall metrics to save computation time
        if phase == "train" and not (hasattr(self, 'epochs') and self.current_epoch == self.epochs - 1):
            # Handle case where predictions is a 2D array
            if predictions.ndim > 1:
                predictions = predictions.flatten()

            # Handle case where true_labels is a 2D array
            if true_labels.ndim > 1:
                true_labels = true_labels.flatten()

            # Basic metrics with zero_division=0 to avoid errors
            return {
                'accuracy': np.mean(true_labels == predictions),
                'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
                'recall': recall_score(true_labels, predictions, average='macro', zero_division=0),
                'f1': f1_score(true_labels, predictions, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
            }

        # For validation and test phases, calculate the full detailed metrics
        try:
            metrics_dict = Metrics.calculate_metrics(
                true_labels, predictions, probabilities)
        except Exception as e:
            # Fallback to basic metrics if calculation fails
            logging.warning(
                f"Error calculating detailed metrics: {e}. Using basic metrics instead.")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Flatten arrays if needed
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            if true_labels.ndim > 1:
                true_labels = true_labels.flatten()

            return {
                'accuracy': np.mean(true_labels == predictions),
                'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
                'recall': recall_score(true_labels, predictions, average='macro', zero_division=0),
                'f1': f1_score(true_labels, predictions, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
            }

        # Only add per-class metrics on test or final validation if calculation succeeded
        if (phase == "test" or (phase == "val" and hasattr(self, 'current_epoch') and self.current_epoch == self.epochs - 1)) and self.per_class_metrics:
            per_class = {}
            classes = np.unique(true_labels)

            for cls in classes:
                cls_mask = (true_labels == cls)
                if sum(cls_mask) == 0:
                    continue

                cls_true = (true_labels == cls).astype(int)
                cls_pred = (predictions == cls).astype(int)

                try:
                    # Use imported functions directly to avoid reference errors
                    cls_metrics = {
                        'accuracy': np.mean(cls_true == cls_pred),
                        'precision': precision_score(cls_true, cls_pred, zero_division=0),
                        'recall': recall_score(cls_true, cls_pred, zero_division=0),
                        'f1': f1_score(cls_true, cls_pred, zero_division=0),
                        'support': np.sum(cls_true)
                    }
                    class_name = self.class_names[cls] if cls < len(
                        self.class_names) else f"Class {cls}"
                    per_class[class_name] = cls_metrics
                except Exception as e:
                    logging.warning(
                        f"Error calculating metrics for class {cls}: {e}")
                    # Create partial metrics if calculation fails
                    cls_metrics = {
                        'accuracy': np.mean(cls_true == cls_pred),
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'support': np.sum(cls_true)
                    }
                    class_name = self.class_names[cls] if cls < len(
                        self.class_names) else f"Class {cls}"
                    per_class[class_name] = cls_metrics

            metrics_dict['per_class'] = per_class

        return metrics_dict

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
        """Create minimal visualizations during training"""
        if not hasattr(self, 'visualization'):
            return

        # Only create visualizations for final epoch or every 10 epochs
        if epoch % 10 != 0 and epoch != self.epochs - 1:
            return

        # Only visualize validation or test phases (skip training phase visualization)
        if phase == "train" and epoch != self.epochs - 1:
            return

        try:
            # Only create visualization if we have per-class metrics (final epoch or test)
            if 'per_class' in metrics:
                self.visualization.visualize_metrics(
                    metrics=metrics,
                    task_name=self.task_name,
                    dataset_name=self.dataset_name,
                    model_name=self.model_name,
                    phase=phase,
                    epoch=epoch,
                    class_names=self.class_names
                )
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
            self.args, 'early_stopping_metric', 'loss')  # Default to loss instead of f1
        saved_attacks = False

        # For adversarial training, initialize tracking of best adversarial metrics
        if self.adversarial:
            self.best_adv_acc = 0.0
            # Consider both clean and adversarial performance for early stopping
            combined_early_stopping = getattr(
                self.args, 'combined_early_stopping', True)
            if combined_early_stopping:
                logging.info(
                    "Using combined clean+adversarial metrics for early stopping")

        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_combined_score = 0.0  # New metric for balancing clean and robust accuracy

        # Log model graph to TensorBoard (try with a sample batch)
        try:
            sample_input = next(iter(self.train_loader))[0][:1].to(self.device)
            self.tb_logger.log_model_graph(self.model, sample_input)
        except Exception as e:
            logging.warning(
                f"Could not add model graph to TensorBoard: {str(e)}")

        # Add tracking for detailed metrics
        MAX_SAFE_EPSILON = 0.025  # Maximum safe epsilon where training is stable
        MIN_ALLOWED_ACCURACY = 0.65  # Minimum allowed validation accuracy

        for epoch in range(self.epochs):
            self.current_epoch = epoch  # Track current epoch for metrics optimization

            # Update epoch for dynamic sample weighting if needed
            if isinstance(self.criterion, DynamicSampleWeightedLoss):
                self.criterion.update_epoch(epoch)
                logging.info(
                    f"Updated dynamic loss function for epoch {epoch+1}")

            # Update adversarial training parameters if using
            if self.adversarial:
                # Store previous epsilon value before updating
                prev_epsilon = getattr(
                    self, 'prev_epsilon', self.adversarial_trainer.initial_epsilon)
                self.prev_epsilon = self.adversarial_trainer.epsilon

                # Update parameters with safety check
                if hasattr(self, 'val_acc') and self.val_acc < MIN_ALLOWED_ACCURACY:
                    # If accuracy is poor, don't increase epsilon and actually reduce it
                    logging.warning(
                        f"Validation accuracy ({self.val_acc:.4f}) below threshold. Reducing epsilon.")
                    self.adversarial_trainer.epsilon = max(
                        self.adversarial_trainer.initial_epsilon,
                        self.adversarial_trainer.epsilon * 0.8  # Reduce by 20%
                    )
                else:
                    # Normal parameter update with safety cap
                    self.adversarial_trainer.update_parameters(epoch)
                    # Cap epsilon to a safe maximum
                    if self.adversarial_trainer.epsilon > MAX_SAFE_EPSILON:
                        logging.warning(
                            f"Capping epsilon at {MAX_SAFE_EPSILON} for stability")
                        self.adversarial_trainer.epsilon = MAX_SAFE_EPSILON

                # Also update attack alpha (step size) to be proportional to epsilon
                self.adversarial_trainer.alpha = min(
                    self.adversarial_trainer.epsilon / 6.0, 0.004)

                logging.info(f"Epoch {epoch+1}: Updated adversarial parameters - "
                             f"epsilon={self.adversarial_trainer.epsilon:.4f}, "
                             f"alpha={self.adversarial_trainer.alpha:.4f}, "
                             f"clean_weight={self.adversarial_trainer.clean_weight:.2f}, "
                             f"adv_weight={self.adversarial_trainer.adv_weight:.2f}")

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
                        data = data.to(self.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.device)

                    if self.adversarial and not saved_attacks and epoch == 0 and batch_idx == 0:
                        self.adversarial_trainer.save_attack_samples(
                            data, data)
                        saved_attacks = True

                    with autocast():
                        output = self.model(data)
                        clean_loss = self.criterion(output, target)

                        # Add adversarial loss if enabled
                        if self.adversarial:
                            # Get batch indices for pregenerated attacks if available
                            batch_indices = list(range(batch_idx * len(data),
                                                       (batch_idx + 1) * len(data)))

                            # Calculate adversarial loss with current parameters
                            adv_loss = self.adversarial_trainer.adversarial_loss(
                                data, target, batch_indices)

                            # Use the current weights that add up to 1.0
                            combined_loss = (self.adversarial_trainer.clean_weight * clean_loss +
                                             self.adversarial_trainer.adv_weight * adv_loss)
                            loss = combined_loss
                            adv_loss_sum += adv_loss.item()

                            # Calculate adversarial accuracy for logging
                            with torch.no_grad():
                                # Generate adversarial examples
                                if hasattr(self.adversarial_trainer.attack, 'generate'):
                                    adv_data = self.adversarial_trainer.attack.generate(
                                        data, target, epsilon=self.adversarial_trainer.epsilon)
                                else:
                                    _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                        data, target)

                                # Calculate adversarial accuracy
                                adv_output = self.model(adv_data)
                                adv_pred = adv_output.argmax(dim=1)
                                adv_correct += (adv_pred ==
                                                target).sum().item()
                        else:
                            loss = clean_loss

                    if not torch.isfinite(loss):
                        logging.warning(
                            f"Non-finite loss detected in batch {batch_idx}")
                        continue

                    self.scaler.scale(loss).backward()
                    with torch.no_grad():
                        # Get predictions and update tracking metrics
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        batch_loss = loss.item()
                        epoch_loss += batch_loss

                        # Store predictions and labels for metrics calculation
                        epoch_true_labels.extend(target.cpu().numpy())
                        epoch_predictions.extend(pred.cpu().numpy())

                        # Store probabilities for metrics calculation
                        probs = torch.nn.functional.softmax(output, dim=1)
                        epoch_probabilities.extend(probs.cpu().numpy())

                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        # Unscale before clipping
                        self.scaler.unscale_(self.optimizer)

                        # Apply gradient clipping if configured
                        if hasattr(self.args, 'max_grad_norm'):
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.max_grad_norm)

                        # Step optimizer and update scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    if batch_idx % 100 == 0:
                        accuracy = correct / total if total > 0 else 0
                        logging.info(
                            f"Epoch: {epoch+1}/{self.epochs} | Batch: {batch_idx * len(data)}/{len(self.train_loader.dataset)} | Loss: {batch_loss:.4f} | Accuracy: {accuracy:.4f}")

                except RuntimeError as err:
                    logging.error(f"Runtime error in batch {batch_idx}: {err}")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler = GradScaler()  # Reset scaler after error
                    continue

                except Exception as exp:
                    logging.exception(
                        f"Unexpected error in batch {batch_idx}: {exp}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

            # Validate with clean data
            val_loss, val_accuracy, val_detailed_metrics = self.validate_with_metrics()

            # Extract F1 score and balanced accuracy from detailed metrics
            val_f1 = val_detailed_metrics.get('f1', 0.0)
            val_balanced_acc = val_detailed_metrics.get(
                'balanced_accuracy', 0.0)

            # For adversarial models, also evaluate adversarial validation accuracy
            adv_val_loss = float('inf')
            adv_val_accuracy = 0.0

            if self.adversarial:
                adv_val_loss, adv_val_accuracy = self.validate_adversarial()
                logging.info(f"Validation - Clean: Loss={val_loss:.4f}, Acc={val_accuracy:.4f} | "
                             f"Adversarial: Loss={adv_val_loss:.4f}, Acc={adv_val_accuracy:.4f}")

                # Calculate a combined score that balances clean and robust performance
                # This weighted harmonic mean penalizes large gaps between clean and robust accuracy
                combined_score = 2 * val_accuracy * adv_val_accuracy / \
                    (val_accuracy + adv_val_accuracy + 1e-10)
                logging.info(
                    f"Combined clean-robust score: {combined_score:.4f}")

            # Apply scheduler after validation if using ReduceLROnPlateau
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.adversarial and combined_early_stopping:
                        # Use combined score for ReduceLROnPlateau
                        self.scheduler.step(combined_score)
                    elif early_stopping_metric == 'f1':
                        self.scheduler.step(val_f1)
                    elif early_stopping_metric == 'balanced_acc':
                        self.scheduler.step(val_balanced_acc)
                    else:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                self.current_lr = self.optimizer.param_groups[0]['lr']

            # Save best model based on validation performance using the selected metric
            improved = False

            if self.adversarial and combined_early_stopping:
                # For adversarial training, use combined score for early stopping
                if combined_score > self.best_combined_score:
                    improved = True
                    self.best_combined_score = combined_score
                    # Update all metrics when the model improves
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_accuracy
                    self.best_val_f1 = val_f1
                    self.best_val_balanced_acc = val_balanced_acc
            else:
                # Traditional early stopping based on selected metric
                if early_stopping_metric == 'loss':
                    if val_loss < self.best_val_loss:
                        improved = True
                        # Update all metrics when the model improves
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_accuracy  
                        self.best_val_f1 = val_f1
                        self.best_val_balanced_acc = val_balanced_acc
                elif early_stopping_metric == 'f1':
                    if val_f1 > self.best_val_f1:
                        improved = True
                        # Update all metrics when the model improves
                        self.best_val_f1 = val_f1
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_accuracy
                        self.best_val_balanced_acc = val_balanced_acc
                elif early_stopping_metric == 'balanced_acc':
                    if val_balanced_acc > self.best_val_balanced_acc:
                        improved = True
                        # Update all metrics when the model improves
                        self.best_val_balanced_acc = val_balanced_acc
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_accuracy
                        self.best_val_f1 = val_f1
                else:
                    if val_accuracy > self.best_val_acc:
                        improved = True
                        # Update all metrics when the model improves
                        self.best_val_acc = val_accuracy
                        self.best_val_loss = val_loss
                        self.best_val_f1 = val_f1
                        self.best_val_balanced_acc = val_balanced_acc

            if improved:
                self.no_improvement_count = 0
                self.save_model(
                    f"save_model/best_{self.model_name}_{self.dataset_name}.pth")
                if self.adversarial and combined_early_stopping:
                    logging.info(
                        f"Improved combined score to {self.best_combined_score:.4f}! Saving model.")
                else:
                    logging.info(
                        f"Improved {early_stopping_metric}! Saving model.")
            else:
                self.no_improvement_count += 1
                logging.info(
                    f"No improvement in {early_stopping_metric if not (self.adversarial and combined_early_stopping) else 'combined score'} "
                    f"for {self.no_improvement_count} epochs.")

            # Apply early stopping after minimum epochs
            if self.no_improvement_count >= patience and epoch >= min_epochs:
                logging.info(
                    f"Early stopping triggered after {epoch+1} epochs")
                break

            # Calculate and log epoch metrics
            epoch_acc = correct / total if total > 0 else 0
            adv_accuracy = (
                adv_correct / total) if (self.adversarial and total > 0) else 0
            avg_adv_loss = (adv_loss_sum / len(self.train_loader)
                            ) if self.adversarial else 0

            # Update adversarial metrics tracking
            self.adv_metrics.update_adversarial_comparison(
                phase='train',
                clean_loss=epoch_loss / len(self.train_loader),
                clean_acc=epoch_acc,
                adv_loss=avg_adv_loss,
                adv_acc=adv_accuracy
            )

            if self.adversarial:
                logging.info(
                    f'Epoch {epoch+1} Training - Clean: Loss={epoch_loss/len(self.train_loader):.4f}, Acc={epoch_acc:.4f} | '
                    f'Adversarial: Loss={avg_adv_loss:.4f}, Acc={adv_accuracy:.4f} | '
                    f'Weights: Clean={self.adversarial_trainer.clean_weight:.2f}, Adv={self.adversarial_trainer.adv_weight:.2f}'
                )
            else:
                logging.info(
                    f'Epoch {epoch+1} Training - Loss={epoch_loss/len(self.train_loader):.4f}, Acc={epoch_acc:.4f}')

            # Update history and visualize adversarial metrics
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

            # After validation, store values for stability check
            self.prev_val_acc = val_accuracy
            self.prev_adv_val_acc = adv_val_accuracy if self.adversarial else 0

            # Add NaN detection and recovery mechanism
            if self.adversarial and (np.isnan(val_loss) or np.isnan(adv_val_loss)):
                logging.warning(
                    f"NaN loss detected in epoch {epoch+1}! Taking recovery actions...")

                # 1. Roll back epsilon to a safe value
                self.adversarial_trainer.epsilon = max(
                    self.adversarial_trainer.initial_epsilon,
                    self.adversarial_trainer.epsilon * 0.5  # Cut epsilon in half
                )

                # 2. Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    self.current_lr = param_group['lr']
                    logging.info(f"Reduced learning rate to {self.current_lr}")

                # 3. Reload weights from last good checkpoint if available
                last_good_path = os.path.join(
                    'out', self.task_name, self.dataset_name, self.model_name,
                    'adv' if self.adversarial else '',
                    'save_model', f'best_{self.model_name}_{self.dataset_name}.pth'
                )
                if os.path.exists(last_good_path):
                    try:
                        logging.info(
                            f"Loading last good checkpoint from {last_good_path}")
                        self.model.load_state_dict(torch.load(last_good_path))
                    except Exception as e:
                        logging.error(f"Failed to load checkpoint: {e}")

                # 4. Reset optimizer state
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler = GradScaler()  # Reset scaler

                # If we've had multiple NaN events, consider early stopping
                if getattr(self, 'nan_count', 0) >= 2:
                    logging.warning(
                        "Multiple NaN events detected. Triggering early stopping.")
                    break
                else:
                    self.nan_count = getattr(self, 'nan_count', 0) + 1

        # After training complete - REMOVE automatic test that causes duplication
        # Instead just log best metrics
        logging.info(f"Training finished. Best metrics - "
                     f"Loss: {self.best_val_loss:.4f}, "
                     f"Accuracy: {self.best_val_acc:.4f}, "
                     f"F1: {self.best_val_f1:.4f}, "
                     f"Balanced Acc: {self.best_val_balanced_acc:.4f}")

        # Store the test metrics properties for access after training
        self.best_metrics = {
            'loss': self.best_val_loss,
            'accuracy': self.best_val_acc,
            'f1': self.best_val_f1,
            'balanced_acc': self.best_val_balanced_acc
        }

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

            # Visualize metrics only on milestone epochs
            if hasattr(self, 'current_epoch') and (self.current_epoch % 10 == 0 or self.current_epoch == self.epochs - 1):
                # Log detailed metrics
                logging.info(f"Validation - Loss: {val_loss:.4f}, "
                             f"Accuracy: {accuracy:.4f}, "
                             f"Precision: {detailed_metrics['precision']:.4f}, "
                             f"Recall: {detailed_metrics['recall']:.4f}, "
                             f"F1: {detailed_metrics['f1']:.4f}")

                # Only log per-class metrics on final epoch to reduce verbosity
                if 'per_class' in detailed_metrics and self.current_epoch == self.epochs - 1:
                    per_class = detailed_metrics['per_class']
                    for cls_name, cls_metrics in per_class.items():
                        logging.info(f"  {cls_name}: F1={cls_metrics['f1']:.4f}, "
                                     f"Precision={cls_metrics['precision']:.4f}, "
                                     f"Recall={cls_metrics['recall']:.4f}")

                # Visualize metrics only on final epoch
                if self.current_epoch == self.epochs - 1:
                    self._visualize_metrics(
                        detailed_metrics, epoch=self.current_epoch, phase="val")
            else:
                # Simple log for non-milestone epochs
                logging.info(
                    f"Validation - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            return val_loss, accuracy, detailed_metrics

        except Exception as e:
            logging.error(f"Error during validation with metrics: {e}")
            return float('inf'), 0.0, {}
        finally:
            self.model.train()

    def validate_adversarial(self):
        """Specific validation method for adversarial examples"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # Add more detailed logging
        logging.info(
            f"Running adversarial validation with epsilon={self.adversarial_trainer.epsilon:.4f}")

        try:
            # Track original performance first
            orig_val_loss, orig_accuracy = self.validate()

            # If original validation accuracy is already poor, don't even attempt adversarial validation
            # as this could make training unstable
            if orig_accuracy < 0.5:
                logging.warning(
                    f"Clean accuracy too low ({orig_accuracy:.4f}), skipping adversarial validation")
                return float('inf'), 0.0

            for batch_idx, (data, target) in enumerate(self.val_loader):
                if isinstance(data, torch.Tensor):
                    data = data.to(self.device, non_blocking=True)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device, non_blocking=True)

                # Generate adversarial examples for validation with error handling
                with torch.enable_grad():  # Need gradients for attack generation
                    try:
                        if hasattr(self.adversarial_trainer.attack, 'generate'):
                            adv_data = self.adversarial_trainer.attack.generate(
                                data, target, self.adversarial_trainer.epsilon)
                        else:
                            # Use a smaller epsilon for validation if regular epsilon is large
                            safe_epsilon = min(
                                self.adversarial_trainer.epsilon, 0.025)
                            _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                data, target, epsilon=safe_epsilon)

                        # Validate the adversarial examples aren't garbage
                        if torch.isnan(adv_data).any():
                            logging.warning(
                                "NaN values detected in adversarial examples")
                            adv_data = data.clone()  # Fallback to clean data

                    except Exception as e:
                        logging.error(
                            f"Error generating adversarial examples: {e}")
                        adv_data = data.clone()  # Fallback to clean data

                # Evaluate on adversarial examples
                with torch.no_grad(), autocast():
                    output = self.model(adv_data)
                    loss = self.criterion(output, target)

                    # Check for NaN and replace with high loss value
                    if torch.isnan(loss):
                        loss = torch.tensor(1000.0, device=self.device)
                        logging.warning(
                            "NaN loss in adversarial validation, replacing with high loss value")

                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            val_loss /= len(self.val_loader)
            accuracy = correct / total if total > 0 else 0

            # Store as instance attributes for stability checks
            self.adv_val_loss = val_loss
            self.adv_val_acc = accuracy

            return val_loss, accuracy

        except Exception as e:
            logging.error(f"Error during adversarial validation: {e}")
            return float('inf'), 0.0
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

        test_loss /= len(self.test_loader)
        accuracy = correct / total if total > 0 else 0

        # Store these metrics as class attributes for easier access later
        self.test_loss = test_loss
        self.test_accuracy = accuracy

        # Calculate and log detailed metrics
        detailed_metrics = self.calculate_detailed_metrics(
            np.array(self.true_labels),
            np.array(self.predictions),
            np.array(self.probabilities),
            phase="test"
        )

        self.test_f1 = detailed_metrics.get('f1', 0.0)

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
        if not hasattr(self, 'visualization'):
            return 0.5  # Default threshold

        # Use the consolidated visualization class method
        return self.visualization.create_threshold_curve(
            true_labels=true_labels,
            probabilities=probabilities,
            task_name=self.task_name,
            dataset_name=self.dataset_name,
            model_name=self.model_name
        )

    def save_model(self, path):
        filename, ext = os.path.splitext(path)
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{filename}_epochs{self.epochs}_lr{self.args.lr}_batch{self.args.train_batch}_{timestamp}{ext}"

        # Create the proper directory structure based on adversarial flag
        if self.adversarial:
            # Create full path including the 'save_model' subdirectory
            dir_path = os.path.join(
                'out', self.task_name, self.dataset_name, self.model_name, 'adv', 'save_model')
            full_path = os.path.join(dir_path, os.path.basename(filename))
        else:
            dir_path = os.path.join(
                'out', self.task_name, self.dataset_name, self.model_name, 'save_model')
            full_path = os.path.join(dir_path, os.path.basename(filename))

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the model
        torch.save(self.model.state_dict(), full_path)
        logging.info(f'Model saved to {full_path}')

        # Also save as latest.pth for recovery purposes
        latest_path = os.path.join(
            dir_path, f"latest_{self.model_name}_{self.dataset_name}.pth")
        torch.save(self.model.state_dict(), latest_path)

        # For adversarial models, also save a checkpoint at each epsilon value for possible rollback
        if self.adversarial and hasattr(self, 'adversarial_trainer'):
            epsilon_str = f"{self.adversarial_trainer.epsilon:.4f}".replace(
                '.', '_')
            epsilon_path = os.path.join(
                dir_path, f"eps_{epsilon_str}_{self.model_name}_{self.dataset_name}.pth")
            torch.save(self.model.state_dict(), epsilon_path)
            logging.info(f'Saved epsilon checkpoint at {epsilon_path}')

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

        # Create DataFrame with only export-safe columns
        data = {'epoch': self.history['epoch']}
        for k in keys_to_check:
            if k in self.history and len(self.history[k]) >= len(self.history['epoch']):
                # Skip complex data types that can't be easily put in a DataFrame
                if k in ['true_labels', 'predictions']:
                    continue
                data[k] = self.history[k]

        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"History saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving history to CSV: {e}")


class TrainingManager:
    """Manages the training process for multiple models and datasets"""

    def __init__(self, args):
        self.args = args
        self._setup_random_seeds(args.manualSeed)

        self.model_loader = ModelLoader(
            args.device, args.arch, pretrained=getattr(args, 'pretrained', True), fp16=getattr(args, 'fp16', False))
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

    def train_dataset(self, dataset_name, run_test=False):
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
                    targets = train_loader.dataset.targets
                    unique, counts = np.unique(targets, return_counts=True)
                    class_counts = dict(zip(unique, counts))
                elif hasattr(train_loader.dataset, 'samples'):
                    counts = {}
                    for _, idx in train_loader.dataset.samples:
                        counts[idx] = counts.get(idx, 0) + 1
                    class_counts = counts
                if class_counts:
                    logging.info(f"Class distribution: {class_counts}")
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
                    dataset_name=dataset_name,
                    adversarial=self.args.adversarial  # Add this parameter
                )

                # Train each model variation
                for model, model_name in models_and_names:
                    # Create optimizer and scheduler for this model
                    try:
                        # Fix: Pass arguments properly for backward compatibility
                        optimizer = self.optimizer_loader.get_optimizer(
                            model=model,
                            optimizer_name=getattr(
                                self.args, 'optimizer', 'adam'),
                            lr=self.args.lr,
                            weight_decay=getattr(
                                self.args, 'weight_decay', 1e-4)
                        )

                        scheduler = None
                        if getattr(self.args, 'scheduler', 'none') != 'none':
                            # FIX: Change scheduler_name to scheduler in the call
                            scheduler = self.lr_scheduler_loader.get_scheduler(
                                optimizer=optimizer,
                                scheduler=self.args.scheduler,  # Changed from scheduler_name to scheduler
                                config=self.args
                            )

                        # Create loss function
                        criterion = nn.CrossEntropyLoss()

                        # Create trainer
                        trainer = Trainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            optimizer=optimizer,
                            criterion=criterion,
                            model_name=model_name,
                            task_name=self.args.task_name,
                            dataset_name=dataset_name,
                            device=self.args.device,
                            config=self.args,
                            scheduler=scheduler
                        )

                        # Train model
                        # Default to 15 epochs patience
                        patience = getattr(self.args, 'patience', 15)
                        trainer.train(patience)

                        # Save training history
                        trainer.save_history_to_csv("training_history.csv")

                        # Optionally run tests after training
                        if run_test:
                            test_results = trainer.test()
                            logging.info(
                                f"Test results for {model_name}: {test_results}")
                    except Exception as e:
                        logging.error(
                            f"Error initializing training components for {model_name}: {str(e)}")
                        continue

            except Exception as e:
                logging.error(
                    f"Error training {arch} on {dataset_name}: {str(e)}")
                continue

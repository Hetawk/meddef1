import argparse
import logging
import os
from datetime import datetime
import random
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from utils.adv_metrics import AdversarialMetrics
from utils.training_logger import TrainingLogger
from utils.robustness.regularization import Regularization
from utils.timer import Timer
from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader
from utils.robustness.optimizers import OptimizerLoader
from utils.robustness.lr_scheduler import LRSchedulerLoader
import json
import warnings  # added import
from tqdm import tqdm   # Added import for progress bar
import torch.backends.cudnn  # Add this to ensure cudnn is recognized

# added to suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')

    # Dataset and processing
    parser.add_argument('--data', nargs='+', required=True, type=str,
                        help='Dataset names to process')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory')

    # Model architecture - update these two arguments
    parser.add_argument('--arch', '-a', nargs='+', default=['meddef', 'resnet', 'densenet'],
                        help='Architecture(s) to use. Provide one or multiple values. Separate multiple names with space or comma.')
    parser.add_argument('--depth', type=str, default='{"meddef": [1.0, 1.1], "resnet": [18, 34], "densenet": [121]}',
                        help='Model depths as JSON string')

    # Training parameters
    parser.add_argument('--train_batch', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=32,
                        help='Test/Val batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--drop', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lambda_l2', type=float, default=1e-4,
                        help='L2 regularization strength')

    # Training control
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm')

    # Device configuration
    parser.add_argument('--gpu-ids', default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--device-index', type=int, default=0,
                        help='Primary GPU index to use')

    # Task specification
    parser.add_argument('--task_name', type=str,
                        choices=['normal_training', 'attack', 'defense'],
                        default='normal_training',
                        help='Task to perform')

    # Add back manualSeed argument
    parser.add_argument('--manualSeed', type=int, default=None,
                        help='manual seed for reproducibility')

    # Add optimizer argument
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adagrad'],
                        help='Optimizer to use for training')

    # Add scheduler argument
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'ExponentialLR',
                                 'ReduceLROnPlateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_step', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')

    # Add adversarial training arguments
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable adversarial training')
    parser.add_argument('--attack_type', type=str, nargs='+', 
                        default=['fgsm'],
                        choices=['fgsm', 'pgd', 'bim', 'jsma'],
                        help='Type(s) of attack for adversarial training. Can specify multiple attacks.')
    parser.add_argument('--attack_eps', type=float, default=0.3,
                        help='Epsilon for adversarial attacks')
    parser.add_argument('--attack_alpha', type=float, default=0.01,
                        help='Alpha for adversarial attacks')
    parser.add_argument('--attack_steps', type=int, default=40,
                        help='Number of steps for iterative attacks')
    parser.add_argument('--adv_weight', type=float, default=1.0,
                        help='Weight for adversarial loss')
    parser.add_argument('--adv_init_mag', type=float, default=0.01,
                        help='Initial magnitude for adversarial perturbation')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Alpha for PGD attack')
    parser.add_argument('--iterations', type=int, default=40,
                        help='Iterations for PGD attack')
    parser.add_argument('--save_attacks', action='store_true',
                        help='Save generated adversarial samples')

    # New arguments for defense:
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to saved model weights')
    parser.add_argument('--prune_rate', type=float, default=0.3,
                        help='Pruning rate for defense task')

    args = parser.parse_args()

    # Process architecture names
    if isinstance(args.arch, str):
        args.arch = [x.strip() for x in args.arch.strip('[]').split(',')]

    # Convert depth string to dictionary - simplified version
    if isinstance(args.depth, str):
        # Remove spaces and single quotes
        depth_str = args.depth.strip()
        if depth_str.startswith("'") and depth_str.endswith("'"):
            depth_str = depth_str[1:-1]

        try:
            # First try direct JSON parsing
            args.depth = json.loads(depth_str)
        except json.JSONDecodeError:
            # If that fails, try manual parsing
            # Remove curly braces
            depth_str = depth_str.strip('{}')
            # Split into key and value
            key, value = depth_str.split(':', 1)
            key = key.strip()
            value = value.strip()
            # Parse the value as a list
            value = eval(value)  # Safe here since we know it's a list
            args.depth = {key: value}

    # Configure CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    use_cuda = torch.cuda.is_available()
    args.device = torch.device(
        f"cuda:{args.device_index}" if use_cuda else "cpu")

    return args


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
        'adv_targets': []
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
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_name = model_name
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.device = device
        self.config = config

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

        self.error_if_nonfinite = False
        self.val_loss = float('inf')
        self.current_lr = self.args.lr
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.no_improvement_count = 0
        self.adv_metrics = AdversarialMetrics()

        # Initialize the lists here for test results
        self.true_labels = []
        self.predictions = []
        self.adv_predictions = []

    @staticmethod
    def _setup_random_seeds(self, seed):
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_model_params(self):
        return sum(p.numel() for p in self.model.parameters()) / 1000000.0

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
        # log_points = [0, total_batches // 2, total_batches - 1]
        initial_params = self.get_model_params()
        logging.info(f"Initial model parameters: {initial_params:.2f}M")

        # No need to store start_time if not used besides logging—included in progress log below.
        saved_attacks = False

        for epoch in range(self.epochs):
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
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_accuracy
                self.no_improvement_count = 0
                self.save_model(
                    f"save_model/best_{self.model_name}_{self.dataset_name}.pth")
            else:
                self.no_improvement_count += 1
            if self.no_improvement_count >= patience:
                logging.info(
                    f"Early stopping triggered after {epoch + 1} epochs")
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
        return self.best_val_loss, self.best_val_acc

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

    def test(self):
        self.model.eval()
        test_loss = 0
        adv_test_loss = 0
        correct = 0
        adv_correct = 0
        total = 0
        self.true_labels = []
        self.predictions = []
        self.adv_predictions = []
        adv_test_accuracy = 0  # Initialize
        with torch.no_grad():
            for data, target in self.test_loader:
                if isinstance(data, torch.Tensor):
                    data = data.to(self.device)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if self.adversarial:
                    with torch.enable_grad():
                        if hasattr(self.adversarial_trainer.attack, 'generate'):
                            adv_data = self.adversarial_trainer.attack.generate(
                                data, target, self.args.epsilon)
                        else:
                            _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                data, target)
                    adv_output = self.model(adv_data)
                    adv_test_loss += self.criterion(adv_output, target).item()
                    adv_pred = adv_output.argmax(dim=1, keepdim=True)
                    adv_correct += adv_pred.eq(target.view_as(adv_pred)
                                               ).sum().item()
                    self.adv_predictions.extend(adv_output.cpu().numpy())
                total += target.size(0)
                self.true_labels.extend(target.cpu().numpy())
                self.predictions.extend(output.cpu().numpy())
        test_loss /= len(self.test_loader)
        accuracy = correct / total if total > 0 else 0
        if self.adversarial:
            adv_test_loss /= len(self.test_loader)
            adv_test_accuracy = adv_correct / total if total > 0 else 0
            logging.info(
                f'Test Results - Clean: Loss={test_loss:.4f}, Acc={accuracy:.4f} | Adversarial: Loss={adv_test_loss:.4f}, Acc={adv_test_accuracy:.4f}')
        else:
            logging.info(
                f'Test Results - Loss={test_loss:.4f}, Accuracy={accuracy:.4f}')
        self.model.train()
        return (test_loss, accuracy) if not self.adversarial else (test_loss, accuracy, adv_test_loss, adv_test_accuracy)

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

        keys_to_check = ['loss', 'accuracy', 'duration',
                         'val_loss', 'val_accuracy', 'true_labels', 'predictions']

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
                    trainer = Trainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        criterion=torch.nn.CrossEntropyLoss(),
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
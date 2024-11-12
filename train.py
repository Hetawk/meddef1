import os
import logging
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import random
from torch.cuda.amp import GradScaler, autocast
from gan.defense.adv_train import AdversarialTraining
from training_logger import TrainingLogger
from utils.robustness.regularization import Regularization
from utils.timer import Timer
from utils.algorithms.supervised import SupervisedLearning

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion, model_name, task_name,
                 dataset_name, device, args, attack_loader=None, scheduler=None, cross_validator=None, adversarial=False):
        self.scheduler = scheduler
        self.cross_validator = cross_validator
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_name = task_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.lambda_l2 = args.lambda_l2
        self.dropout_rate = args.drop
        self.alpha = args.alpha
        self.patience = args.patience
        self.attack_loader = attack_loader
        self.timer = Timer()
        self.epochs = args.epochs
        self.history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'duration': [],
            'true_labels': [],
            'predictions': [],
            'val_loss': [],
            'val_accuracy': []
        }

        self.device = device
        self.model.to(self.device)
        self.has_trained = False
        self.args = args

        Regularization.apply_dropout(self.model, self.dropout_rate)

        self.true_labels = []
        self.predictions = []
        self.supervised_learning = SupervisedLearning()
        self.adversarial = adversarial
        if adversarial:
            self.adversarial_training = AdversarialTraining(model, criterion, epsilon=0.3, alpha=0.01)
        # Clear cache before starting
        torch.cuda.empty_cache()
        self.scaler = GradScaler()
        self.accumulation_steps = args.accumulation_steps

        # Set random seed for reproducibility
        self.set_random_seed(args.manualSeed)
        # Initialize TrainingLogger
        self.training_logger = TrainingLogger()


    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, patience, adversarial=False):
        if self.has_trained:
            logging.warning(f"{self.model} has already been trained. Training again will overwrite the existing model.")
            return
        logging.info(f"Training {self.model_name}...")
        self.has_trained = True

        torch.cuda.empty_cache()

        # Set CUDA_LAUNCH_BLOCKING
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(self.device)

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        total_batches = len(self.train_loader)
        log_points = [0, total_batches // 2, total_batches - 1]

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            start_time = datetime.now()
            epoch_true_labels = []
            epoch_predictions = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                data.requires_grad = adversarial

                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.accumulation_steps  # Normalize loss

                if self.adversarial:
                    adv_loss = self.adversarial_training.adversarial_loss(data, target)
                    loss += adv_loss

                l2_reg = Regularization.apply_l2_regularization(self.model, self.lambda_l2,
                                                                log_message=(batch_idx == 0))
                loss = Regularization.integrate_regularization(loss, l2_reg, log_message=(batch_idx == 0))

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.accumulation_steps  # Accumulate the actual loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                epoch_true_labels.extend(target.cpu().numpy())
                epoch_predictions.extend(pred.cpu().numpy())

                if batch_idx in log_points:
                    accuracy = correct / total
                    end_time = datetime.now()
                    duration = Timer.format_duration((end_time - start_time).total_seconds())
                    logging.info(f'Epoch: {epoch + 1}/{self.epochs} '
                                 f'| Batch: {batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                                 f'| Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f} ')

            val_loss, val_accuracy = self.validate()
            accuracy = correct / total
            epoch_loss /= len(self.train_loader)
            end_time = datetime.now()
            epoch_duration = Timer.format_duration((end_time - start_time).total_seconds())
            logging.info(f'Epoch: {epoch + 1}/{self.epochs} | Loss: {epoch_loss:.4f}  | Accu: {accuracy:.4f}  '
                         f'| V_loss: {val_loss:.4f} | V_accu: {val_accuracy:.4f} '
                         f'| Duration: {epoch_duration}')

            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(accuracy)
            self.history['duration'].append(epoch_duration)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['true_labels'].append(epoch_true_labels)
            self.history['predictions'].append(epoch_predictions)

            if epoch_loss < best_loss or val_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                self.save_model(f"save_model/best_{self.model_name}_{self.dataset_name}.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if self.cross_validator:
                self.cross_validator.run()

            # Log training information after each epoch
            metrics = {
                'loss': epoch_loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            test_loss, test_accuracy = self.test()
            test_metrics = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.training_logger.log_training_info(self.task_name, self.model_name, self.dataset_name, vars(self.args),
                                                   metrics, start_time, end_time, test_metrics)

        logging.info(f"Finished training {self.model_name}.")
        self.test()
        self.save_history_to_csv("training_history.csv")

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(self.val_loader.dataset)
        accuracy = correct / len(self.val_loader.dataset)
        self.model.train()
        return val_loss, accuracy

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        self.true_labels = []
        self.predictions = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # Use argmax to get predicted class indices
                correct += pred.eq(target.view_as(pred)).sum().item()
                self.true_labels.extend(target.cpu().numpy())
                self.predictions.extend(output.cpu().numpy())  # Store logits

        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        logging.info(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        self.history['true_labels'][-1] = self.true_labels
        self.history['predictions'][-1] = self.predictions  # Store logits, not predicted labels
        self.model.train()
        return test_loss, accuracy

    def save_model(self, path):
        filename, ext = os.path.splitext(path)
        # Include epochs, learning rate, batch size, and timestamp in the filename
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{filename}_epochs{self.epochs}_lr{self.args.lr}_batch{self.args.train_batch}_{timestamp}{ext}"
        path = os.path.join('out', self.task_name, self.dataset_name, self.model_name, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f'Model saved to {path}')

    def save_history_to_csv(self, filename):
        filename = os.path.join('out', self.task_name, self.dataset_name, self.model_name, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        keys_to_check = ['loss', 'accuracy', 'duration', 'val_loss', 'val_accuracy', 'true_labels', 'predictions']
        for key in keys_to_check:
            if len(self.history[key]) != len(self.history['epoch']):
                raise ValueError(
                    f"Length of {key} ({len(self.history[key])}) does not match length of 'epoch' ({len(self.history['epoch'])}).")

        if len(self.history['true_labels']) != len(self.history['predictions']):
            raise ValueError(
                f"Length of true_labels ({len(self.history['true_labels'])}) does not match length of predictions ({len(self.history['predictions'])}).")

        self.history['model_name'] = [self.model_name] * len(self.history['epoch'])

        history_df = pd.DataFrame(self.history)
        history_df['true_labels'] = history_df['true_labels'].apply(lambda x: ','.join(map(str, x)))
        history_df['predictions'] = history_df['predictions'].apply(lambda x: ','.join(map(str, x)))

        if not os.path.isfile(filename):
            history_df.to_csv(filename, index=False)
        else:
            history_df.to_csv(filename, mode='a', index=False, header=False)

        logging.info(f'Training history saved to {filename}')

    def get_test_results(self):
        # Convert true_labels and predictions to NumPy arrays
        return np.array(self.true_labels), np.array(self.predictions)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Loaded model from {path}")

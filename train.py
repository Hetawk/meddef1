# train.py


import os
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from utils.robustness.regularization import Regularization
from utils.timer import Timer


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion, model_name, task_name,
                 dataset_name,device, lambda_l2=0.0, dropout_rate=0.5, alpha=0.01, attack_loader=None, scheduler=None,
                 cross_validator=None):
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
        self.lambda_l2 = lambda_l2
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.attack_loader = attack_loader
        self.timer = Timer()
        self.epochs = 0
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
        self.model.to(self.device)  # Move model to GPU if available
        self.has_trained = False

        # Ensure dropout is applied at initialization
        Regularization.apply_dropout(self.model, self.dropout_rate)

        # Initialize attributes for storing results
        self.true_labels = []
        self.predictions = []

    def train(self, epochs, patience=5, adversarial=False):
        if self.has_trained:
            logging.warning(f"{self.model} has already been trained. Training again will overwrite the existing model.")
            return
        logging.info(f"Training {self.model_name}...")
        self.has_trained = True

        self.epochs = epochs
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        total_batches = len(self.train_loader)
        log_points = [0, total_batches // 2, total_batches - 1]

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            start_time = datetime.now()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                data.requires_grad = adversarial

                output = self.model(data)
                loss = self.criterion(output, target)

                if adversarial:
                    self.model.zero_grad()
                    loss.backward(retain_graph=True)
                    perturbed_data = data + self.alpha * data.grad.sign()
                    perturbed_data = torch.clamp(perturbed_data, 0, 1)
                    adv_output = self.model(perturbed_data)
                    adv_loss = self.criterion(adv_output, target)
                    if not torch.isnan(adv_loss):
                        loss += adv_loss

                # Apply L2 regularization
                l2_reg = Regularization.apply_l2_regularization(self.model, self.lambda_l2,
                                                                log_message=(batch_idx == 0))
                loss = Regularization.integrate_regularization(loss, l2_reg, log_message=(batch_idx == 0))

                self.optimizer.zero_grad()
                loss.backward()
                # Clip the gradient norm to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                if batch_idx in log_points:
                    accuracy = correct / total
                    end_time = datetime.now()
                    duration = Timer.format_duration((end_time - start_time).total_seconds())
                    logging.info(f'Epoch: {epoch + 1}/{self.epochs}, '
                                 f'Batch: {batch_idx * len(data)}/{len(self.train_loader.dataset)}, '
                                 f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, '
                                 f'Duration: {duration} ')

            val_loss, val_accuracy = self.validate()
            accuracy = correct / total
            epoch_loss /= len(self.train_loader)  # Calculate average loss for the epoch
            end_time = datetime.now()
            epoch_duration = Timer.format_duration((end_time - start_time).total_seconds())
            logging.info(f'Epoch: {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, '
                         f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f},'
                         f'Duration: {epoch_duration}')

            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(accuracy)
            self.history['duration'].append(epoch_duration)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            self.test()

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
                self.scheduler.step(
                    val_loss if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

        self.save_history_to_csv("training_history.csv")

    def cross_validate(self, num_folds=5):
        if self.cross_validator is not None:
            cross_validator = self.cross_validator(self.train_loader.dataset, self.model, self.criterion,
                                                   self.optimizer, num_folds=num_folds)
            cross_validator.run()

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
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                self.true_labels.extend(target.cpu().numpy())
                self.predictions.extend(pred.cpu().numpy())
        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        logging.info(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        # Convert lists to numpy arrays before appending
        self.history['true_labels'].append(np.array(self.true_labels))
        # Convert predictions to a single string representation
        predictions_str = np.array2string(np.array(self.predictions), separator=',')[1:-1]  # Remove the square brackets
        self.history['predictions'].append(predictions_str)
        self.model.train()

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

    def save_model(self, path):
        path = os.path.join("out", self.task_name, self.dataset_name, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f'Model saved to {path}')

    def save_history_to_csv(self, filename):
        filename = os.path.join("out",  self.task_name, self.dataset_name, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Ensure all history lists have the same length
        if not all(len(self.history[key]) == len(self.history['epoch']) for key in
                   ['loss', 'accuracy', 'duration']):
            raise ValueError("Lengths of history lists are not consistent.")
        # Add model_name to history for each epoch
        self.history['model_name'] = [self.model_name] * len(self.history['epoch'])
        # Create DataFrame
        df = pd.DataFrame(self.history)
        # Write DataFrame to CSV
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)  # Write with header if file does not exist
        else:
            df.to_csv(filename, mode='a', index=False, header=False)  # Append without header if file exists
        logging.info(f'Training history saved to {filename}')

    def get_test_results(self):
        return self.true_labels, self.predictions

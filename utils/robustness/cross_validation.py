# utils/robustness/cross_validation.py

import logging
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

from train import Trainer



class CrossValidator:
    def __init__(self, dataset, model, model_name, dataset_name, criterion, optimizer_class, optimizer_params, hyperparams, num_folds=5,
                 num_classes=None, device=None, args=None, attack_loader=None, scheduler=None, cross_validator=None):
        self.dataset = dataset
        self.model = model
        self.model_name = model_name  # Store the model name
        self.dataset_name = dataset_name  # Store the dataset name
        self.criterion = criterion
        self.optimizer_class = optimizer_class  # Store the optimizer class
        self.optimizer_params = optimizer_params  # Store the optimizer parameters
        self.num_folds = num_folds
        self.batch_size = hyperparams['batch_size']
        self.num_epochs = hyperparams['epochs']
        self.num_classes = num_classes
        self.device = device
        self.args = args
        self.attack_loader = attack_loader
        self.scheduler = scheduler
        self.cross_validator = cross_validator
        self.logger = logging.getLogger(__name__)
        self.cross_validation_logged = False

    def run(self):
        if not self.args.use_cross_validator:
            if not self.cross_validation_logged:  # Check if the message has already been logged
                self.logger.info('Cross-Validation is disabled. Skipping...')
                self.cross_validation_logged = True  # Set the flag to True after logging the message
            return

        kfold = KFold(n_splits=self.num_folds, shuffle=True)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            self.logger.info(f'Fold {fold + 1}/{self.num_folds}')
            train_subsampler = Subset(self.dataset, train_ids)
            val_subsampler = Subset(self.dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=self.batch_size, shuffle=False)

            model = self.model(num_classes=self.num_classes).to(self.device)

            # Remove 'momentum' if the optimizer does not support it
            optimizer_params = self.optimizer_params.copy()
            if 'momentum' in optimizer_params and self.optimizer_class not in [optim.SGD, optim.RMSprop]:
                optimizer_params.pop('momentum')

            optimizer = self.optimizer_class(model.parameters(), **optimizer_params)  # Instantiate the optimizer

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=val_loader,
                optimizer=optimizer,
                criterion=self.criterion,
                model_name=self.model_name,  # Use the actual model name
                task_name=self.args.task_name,  # Use the actual task name
                dataset_name=self.dataset_name,  # Use the actual dataset name
                device=self.device,
                args=self.args,
                attack_loader=self.attack_loader,
                scheduler=self.scheduler,
                cross_validator=self.cross_validator
            )
            trainer.train(patience=self.args.patience)
            val_loss, val_accuracy = trainer.validate()
            self.logger.info(f'Fold {fold + 1} - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
            fold_results.append((val_loss, val_accuracy))

        avg_loss = np.mean([result[0] for result in fold_results])
        avg_accuracy = np.mean([result[1] for result in fold_results])
        self.logger.info(
            f'Cross-Validation Results - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

        return fold_results, avg_loss, avg_accuracy

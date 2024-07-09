# cross_validator.py

import logging
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

from train import Trainer


class CrossValidator:
    def __init__(self, dataset, model, criterion, optimizer, hyperparams, num_folds=5):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_folds = num_folds
        self.batch_size = hyperparams['batch_size']
        self.num_epochs = hyperparams['epochs']
        self.logger = logging.getLogger(__name__)

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kfold = KFold(n_splits=self.num_folds, shuffle=True)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            self.logger.info(f'Fold {fold + 1}/{self.num_folds}')
            train_subsampler = Subset(self.dataset, train_ids)
            val_subsampler = Subset(self.dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=self.batch_size, shuffle=False)

            model = self.model().to(device)
            optimizer = self.optimizer(model.parameters())

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=val_loader,
                optimizer=optimizer,
                criterion=self.criterion,
                model_name='cross_val_model',
                task_name='cross_val_task',
                dataset_name='cross_val_dataset'
            )
            trainer.train(self.num_epochs)
            val_loss, val_accuracy = trainer.validate()
            self.logger.info(f'Fold {fold + 1} - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
            fold_results.append((val_loss, val_accuracy))

        avg_loss = np.mean([result[0] for result in fold_results])
        avg_accuracy = np.mean([result[1] for result in fold_results])
        self.logger.info(f'Cross-Validation Results - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

        return fold_results, avg_loss, avg_accuracy

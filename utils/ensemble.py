# ensemble.py
import os

import numpy as np
import pandas as pd
import torch
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


class Ensemble:
    def __init__(self, models, model_names, dataset_name, train_dataset, test_dataset, task_name):
        self.models = models
        self.model_names = model_names
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.task_name = task_name

    def predict(self, loader):
        all_preds = []

        for model in self.models:
            model.eval()
            model.to(next(model.parameters()).device)  # Move model to the same device as the data
            preds = []

            with torch.no_grad():
                for data, _ in loader:
                    data = data.to(next(model.parameters()).device)  # Move data to the same device as the model
                    output = model(data)
                    preds.append(output)

            all_preds.append(torch.cat(preds))

        avg_preds = torch.mean(torch.stack(all_preds), dim=0)
        _, predicted = torch.max(avg_preds, 1)
        return predicted.cpu().numpy()  # Convert tensor to numpy array

    def save_models(self, path, dataset_name):
        for model, model_name in zip(self.models, self.model_names):
            model_path = os.path.join(path, f"{dataset_name}_{model_name}.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logging.info(f'Model {model_name} saved to {model_path}')

    def save_predictions(self, predictions, model_name, accuracy, precision, recall, f1_score, auc_roc, path,
                         dataset_name):
        # Check if predictions is a list or a numpy array
        if not isinstance(predictions, (list, np.ndarray)):
            raise ValueError("Predictions must be a list or a numpy array.")

        # Check if performance metrics are numbers
        metrics = [accuracy, precision, recall, f1_score, auc_roc]
        if not all(isinstance(metric, (int, float)) for metric in metrics):
            raise ValueError("Performance metrics must be numbers.")

        predictions_csv_path = os.path.join(path, f"{dataset_name}_ensemble_predictions.csv")
        os.makedirs(os.path.dirname(predictions_csv_path), exist_ok=True)

        # Create a DataFrame with the new data
        new_df = pd.DataFrame({
            'Model Name': model_name,
            'Predictions': predictions,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'AUC-ROC': auc_roc
        })

        # Check if the file already exists
        if os.path.isfile(predictions_csv_path):
            df = pd.read_csv(predictions_csv_path)
            df = pd.concat([df, new_df])
        else:
            df = new_df

        df.to_csv(predictions_csv_path, index=False)
        logging.info(f"Ensemble predictions saved to {predictions_csv_path}")

    def predict_with_ensemble(self):
        # Use the DataLoader for the test data
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False)

        # Make predictions using the ensemble
        ensemble_predictions = self.predict(test_loader)

        # Check if ensemble_predictions is a list or a numpy array
        if not isinstance(ensemble_predictions, (list, np.ndarray)):
            raise TypeError("ensemble_predictions must be a list or a numpy array.")

        # Get the true labels of the test data
        true_labels = []
        for _, label in test_loader:
            true_labels.extend(label.tolist())

        # Calculate performance metrics
        accuracy = accuracy_score(true_labels, ensemble_predictions)
        precision = precision_score(true_labels, ensemble_predictions, average='macro')
        recall = recall_score(true_labels, ensemble_predictions, average='macro')
        f1 = f1_score(true_labels, ensemble_predictions, average='macro')
        auc_roc = roc_auc_score(true_labels, ensemble_predictions, multi_class='ovr')

        # Save the ensemble predictions
        self.save_predictions(ensemble_predictions, 'ensemble', accuracy, precision, recall, f1, auc_roc,
                              os.path.join('out', self.task_name, self.dataset_name), self.dataset_name)

        # Convert ensemble_predictions and true_labels to one-hot encoded labels
        encoder = OneHotEncoder()
        ensemble_predictions_one_hot = encoder.fit_transform(ensemble_predictions.reshape(-1, 1)).toarray()
        true_labels_one_hot = encoder.transform(np.array(true_labels).reshape(-1, 1)).toarray()

        return true_labels_one_hot, ensemble_predictions_one_hot

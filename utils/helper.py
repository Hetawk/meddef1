# helper.py
import pandas as pd
import logging
import torch
import os

from gan.attack.attack_loader import AttackLoader
from gan.defense.defense_loader import DefenseLoader
from loader.dataset_loader import DatasetLoader
from loader.preprocess import Preprocessor
from train import Trainer
from utils.ensemble import Ensemble
from utils.evaluator import Evaluator
from utils.robustness.cross_validation import CrossValidator
from utils.visual.normal_visual import visualize_all
from utils.visual.visualization import Visualization


class TaskHandler:
    def __init__(self, datasets_dict, models_loader, optimizers_dict, hyperparams_dict, input_channels_dict, classes,
                 dataset_name, lr_scheduler_loader=None, cross_validator=None, device=None):
        self.classes = classes
        self.dataset_name = dataset_name
        self.datasets_dict = datasets_dict
        self.current_task = None
        self.models_loader = models_loader
        self.optimizers_dict = optimizers_dict
        self.lr_scheduler_loader = lr_scheduler_loader
        self.cross_validator = cross_validator
        self.hyperparams_dict = hyperparams_dict
        self.input_channels_dict = input_channels_dict
        self.visualization = Visualization()
        self.attack_loader = None
        self.defense_loader = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def run_train(self):
        """Runs the training process for all datasets and models."""
        self.current_task = 'normal_training'
        logging.info("Training task selected.")
        all_results = []

        for dataset_name, dataset_loader in self.datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")
            train_dataset, val_dataset, test_dataset = dataset_loader.load()
            input_channels = self.input_channels_dict.get(dataset_name)
            logging.info(f"Input channels for {dataset_name}: {input_channels}")

            model_list = []
            model_names_list = []
            true_labels_dict = {}
            predictions_dict = {}

            evaluator = None
            for model_name in self.models_loader.models_dict.keys():
                logging.info(f"Loading model: {model_name}")
                model, trainer = self.train_and_evaluate_model(model_name, dataset_name,
                                                               train_dataset, val_dataset, test_dataset,
                                                               input_channels)
                model_list.append(model)
                model_names_list.append(model_name)
                true_labels, predictions = trainer.get_test_results()
                true_labels_dict[model_name] = true_labels
                predictions_dict[model_name] = predictions

                evaluator = Evaluator(model_name, [], true_labels, predictions, self.current_task)
                evaluator.evaluate(dataset_name)
                all_results.append(evaluator)

            # dataset_loader = DatasetLoader(self.dataset_name)
            # class_names = dataset_loader.get_and_print_classes()
            # self.visualization.visualize_normal(model_names_list, (true_labels_dict, predictions_dict),
            #                                     self.current_task, dataset_name, class_names)
            # visualize_all(model_names_list, (true_labels_dict, predictions_dict), self.current_task, dataset_name,
            #               class_names)
            if evaluator is not None:
                all_results.extend(evaluator.results)

    def run_attack(self):
        self.current_task = 'attack'
        logging.info("Attack task selected.")
        all_results = []

        for dataset_name, dataset_loader in self.datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")
            _, test_dataset, _ = dataset_loader.load()

            model_names = [] # Empty list to store our models
            batch_size = self.hyperparams_dict[dataset_name]['batch_size']

            for model_name in self.models_loader.models_dict.keys():
                logging.info(f"Loading model: {model_name}")
                # Load the pre-trained model from the normal_training task
                model = self.models_loader.load_pretrained_model(model_name, 'normal_training', self.dataset_name)

                model_names.append(model_name)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                model.to(self.device)

                # Assign the new AttackLoader instance to self.attack_loader
                self.attack_loader = AttackLoader(model)
                for attack_name in self.attack_loader.attacks_dict.keys():
                    logging.info(f"Running attack: {attack_name}")
                    attack = self.attack_loader.get_attack(attack_name)

                    adv_examples_list = []
                    for batch_idx, (data, labels) in enumerate(test_loader):
                        if batch_idx >= 3:  # Limit to 3 batches to avoid excessive loops
                            break
                        data, labels = data.to(self.device), labels.to(self.device)
                        adv_example = attack.attack(data, labels)  # adv_example can be a tuple of length 2 or 3

                        # Ensure the tensors have the same size
                        min_size = min(len(adv_example[1]), len(data))
                        adv_example = (adv_example[0][:min_size], adv_example[1][:min_size]) + tuple(adv_example[2:])

                        # Check if data and adv_example are not empty
                        if data.size(0) > 0 and adv_example[1].size(0) > 0:
                            # Ensure adv_examples_list is a tuple of length 2 or 3
                            if len(adv_example) in [2, 3]:
                                adv_examples_list.append(
                                    adv_example)  # Store original and adversarial examples as a tuple
                            else:
                                logging.error(
                                    f"Unexpected data format for batch {batch_idx}. Expected a tuple of length 2 or 3.")
                        else:
                            logging.info(f"Skipping batch {batch_idx} due to empty data or adv_example.")

                    # Visualize adversarial examples
                    self.visualization.visualize_attack(adv_examples_list, model_names, self.current_task, dataset_name,
                                                        attack_name)

                    # Log or save adversarial examples as needed
                    logging.info(
                        f"Generated {len(adv_examples_list)} adversarial examples for dataset: {dataset_name} using attack: {attack_name}")

                    # Collect results for saving
                    all_results.append({
                        'dataset_name': dataset_name,
                        'model_name': model_name,
                        'attack_name': attack_name,
                        'adv_examples': adv_examples_list
                    })

        # Save all results into consolidated CSV file
        # self.save_results(all_results)

    def run_defense(self):
        self.current_task = 'defense'
        logging.info("Defense task selected.")
        for dataset_name, dataset_loader in self.datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")
            train_dataset, test_dataset = dataset_loader.load()
            input_channels = self.input_channels_dict[dataset_name]

            preprocessor = Preprocessor(dataset_name, 'base_model_name')
            train_dataset, test_dataset = preprocessor.preprocess(train_dataset, test_dataset)

            for model_name in self.models_loader.models_dict.keys():
                logging.info(f"Loading model: {model_name}")
                model, _ = self.train_and_evaluate_model(model_name, dataset_name, train_dataset, test_dataset,
                                                         input_channels)  # Ensure the model is trained
                self.defense_loader.model = model
                # Attack phase
                attack_loader = AttackLoader(model)
                for attack_name in attack_loader.attacks_dict.keys():
                    logging.info(f"Running attack: {attack_name}")
                    attack = attack_loader.get_attack(attack_name)
                    adv_examples = attack.attack(train_dataset.data, train_dataset.targets)

                    # Defense phase
                    defense_loader = DefenseLoader(model)
                    for defense_name in defense_loader.defenses_dict.keys():
                        logging.info(f"Running defense: {defense_name}")
                        defense = defense_loader.get_defense(defense_name)
                        defended_model = defense.defend(adv_examples, train_dataset.targets)

                        # Train the defended model
                        adv_train_dataset = torch.utils.data.TensorDataset(adv_examples, train_dataset.targets)
                        defended_model, trainer = self.train_and_evaluate_model(defended_model, dataset_name,
                                                                                adv_train_dataset,
                                                                                test_dataset,
                                                                                input_channels,
                                                                                adversarial=True)  # Train the model on adversarial examples

                self.visualization.visualize_defense((trainer.true_labels, trainer.predictions))

    def train_and_evaluate_model(self, model_name, dataset_name, train_dataset, val_dataset, test_dataset,
                                 input_channels, adversarial=False):
        """Trains and evaluates a model."""
        preprocessor = Preprocessor(model_name, dataset_name, task_name=self.current_task)
        train_dataset, val_dataset, test_dataset = preprocessor.preprocess(train_dataset, val_dataset, test_dataset)

        # Use batch_size from hyperparams_dict
        batch_size = self.hyperparams_dict[dataset_name]['batch_size']

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                 shuffle=False) if val_dataset is not None else None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = self.models_loader.get_model(model_name, input_channels=input_channels)
        model.to(self.device)

        hyperparams = self.hyperparams_dict[dataset_name]
        optimizer = self.optimizers_dict.get_optimizer(hyperparams['optimizer'], model.parameters(),
                                                       lr=hyperparams['lr'], momentum=hyperparams.get('momentum', 0))

        if self.current_task in ['defense']:
            adversarial = True

        if self.current_task is None or dataset_name is None:
            raise ValueError("Task name or dataset name is not set.")

        # Initialize scheduler and cross-validator if they exist
        scheduler = self.lr_scheduler_loader.get_scheduler('ReduceLROnPlateau', optimizer, patience=hyperparams[
            'patience']) if self.lr_scheduler_loader else None
        cross_validator = CrossValidator(train_loader.dataset, model, self.criterion, optimizer, hyperparams,
                                         num_folds=5) if self.cross_validator else None

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            torch.nn.CrossEntropyLoss(),
            model_name,
            self.current_task,
            dataset_name,
            device=self.device,
            lambda_l2=hyperparams['lambda_l2'],
            dropout_rate=hyperparams['dropout_rate'],
            alpha=hyperparams.get('alpha', 0.01),
            attack_loader=self.attack_loader,
            scheduler=scheduler,
            cross_validator=cross_validator
        )

        if adversarial:
            # Use adversarial training
            trainer.train_adversarial(epochs=hyperparams['epochs'], patience=hyperparams['patience'])
        else:
            # Use normal training
            trainer.train(epochs=hyperparams['epochs'], patience=hyperparams['patience'])

        trainer.test()
        model.trained = True

        return model, trainer


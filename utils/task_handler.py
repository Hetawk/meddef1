# task_handler.py
import logging
import os
from datetime import datetime

import numpy as np
import torch
from gan.attack.attack_loader import AttackLoader
from gan.defense.defense_loader import DefenseLoader
from gan.defense.prune import Pruner
from loader.dataset_loader import DatasetLoader
from train import Trainer
from utils.ensemble import Ensemble
from utils.evaluator import Evaluator
from utils.robustness.cross_validation import CrossValidator
from utils.visual.normal_visual import visualize_all
from utils.visual.visualization import Visualization


# task_handler.py

class TaskHandler:
    def __init__(self, datasets_dict, models_loader, optimizers_dict, hyperparams_dict, input_channels_dict, classes,
                 dataset_name, lr_scheduler_loader, cross_validator, device, args, num_classes):
        self.classes = classes  # Ensure this is a dictionary
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
        self.args = args
        self.device = args.device
        self.trained_models = {}
        self.num_classes = num_classes

    def run_train(self):
        """Runs the training process for the specified dataset and models."""
        self.current_task = 'normal_training'
        logging.info("Training task selected.")
        all_results = []

        dataset_name = self.dataset_name
        if dataset_name not in self.datasets_dict:
            raise ValueError(f"Dataset {dataset_name} not found in datasets_dict.")

        dataset_loader = self.datasets_dict[dataset_name]
        train_loader, val_loader, test_loader = dataset_loader.load(
            train_batch_size=self.args.train_batch,
            val_batch_size=self.args.test_batch,
            test_batch_size=self.args.test_batch,
            num_workers=self.args.workers,
            pin_memory=self.args.pin_memory
        )
        logging.info(f"Processing dataset: {dataset_name}")
        input_channels = self.input_channels_dict.get(dataset_name)
        logging.info(f"Input channels for {dataset_name}: {input_channels}")

        model_list = []
        model_names_list = []
        true_labels_dict = {}
        predictions_dict = {}

        for model_name in self.args.arch:  # Iterate over the list of architectures
            if isinstance(self.args.depth, dict):
                depths = self.args.depth.get(model_name, [])
                if not depths:
                    logging.warning(f"No depths specified for model {model_name}. Using default depth.")
                    depths = [None]  # or handle accordingly
            else:
                depths = self.args.depth if isinstance(self.args.depth, list) else [self.args.depth]

            models = self.train_and_evaluate_model(
                model_name, dataset_name, train_loader, val_loader, test_loader, input_channels,
                depths,  # Pass depths directly
                self.num_classes, self.classes[dataset_name]
            )

            for depth, (model, trainer) in models.items():
                model_name_with_depth = f"{model_name}{depth}"
                model_list.append(model)
                model_names_list.append(model_name_with_depth)
                true_labels, predictions = trainer.get_test_results()

                # Ensure predictions have the correct shape
                predictions = np.array(predictions)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                if predictions.shape[1] != self.num_classes:
                    raise ValueError(
                        f'Predictions array shape {predictions.shape} is not correct, expected (number of samples, {self.num_classes})')

                true_labels_dict[model_name_with_depth] = true_labels.tolist()
                predictions_dict[model_name_with_depth] = predictions.tolist()

                evaluator = Evaluator(model_name_with_depth, [], true_labels.tolist(), predictions.tolist(),
                                      self.current_task)
                try:
                    evaluator.evaluate(dataset_name)
                    all_results.append(evaluator)
                except Exception as e:
                    logging.error(f"Error evaluating model {model_name_with_depth}: {e}")

        logging.info(f"Taskhandler: - Class names: {self.classes[dataset_name]}")
        self.visualization.visualize_normal(
            model_names_list, (true_labels_dict, predictions_dict), self.current_task, dataset_name,
            self.classes[dataset_name]
        )

    def run_defense(self):
        self.current_task = 'defense'
        logging.info("Defense task selected.")
        all_results = []

        dataset_name = self.dataset_name
        if dataset_name not in self.datasets_dict:
            raise ValueError(f"Dataset {dataset_name} not found in datasets_dict.")

        dataset_loader = self.datasets_dict[dataset_name]
        train_loader, val_loader, test_loader = dataset_loader.load(
            train_batch_size=self.args.train_batch,
            val_batch_size=self.args.test_batch,
            test_batch_size=self.args.test_batch,
            num_workers=self.args.workers,
            pin_memory=self.args.pin_memory
        )
        logging.info(f"Processing dataset: {dataset_name}")
        input_channels = self.input_channels_dict.get(dataset_name)
        logging.info(f"Input channels for {dataset_name}: {input_channels}")

        for model_name in self.args.arch:
            if isinstance(self.args.depth, dict):
                depths = self.args.depth.get(model_name, [])
                if not depths:
                    logging.warning(f"No depths specified for model {model_name}. Using default depth.")
                    depths = [None]
            elif isinstance(self.args.depth, list):
                depths = self.args.depth
            else:
                depths = [self.args.depth]

            for depth in depths:
                if isinstance(depth, dict):
                    raise ValueError(f"Depth for model {model_name} should be an integer, not a dictionary.")
                model, model_name_with_depth = self.models_loader.get_model(
                    model_name, depth=depth, input_channels=input_channels, num_classes=self.num_classes
                )
                model.to(self.device)

                pruner = Pruner(model, self.args.prune_rate)
                pruned_model = pruner.unstructured_prune()

                # Save the pruned model using the proper path convention
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"best_{model_name_with_depth}_{dataset_name}_pruned_{timestamp}.pth.tar"
                path = os.path.join('out', self.current_task, dataset_name, model_name_with_depth, filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                pruner.save_checkpoint({'state_dict': pruned_model.state_dict()}, path)

                logging.info(f"Pruned model {model_name_with_depth} saved successfully to {path}.")

    def run_attack(self):
        self.current_task = 'attack'
        logging.info("Attack task selected.")
        all_results = []

        for dataset_name, (train_loader, val_loader, test_loader) in self.datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")

            model_names = []
            for model_name in self.models_loader.models_dict.keys():
                logging.info(f"Loading model: {model_name}")
                model = self.models_loader.load_pretrained_model(model_name, 'normal_training', dataset_name)
                model.to(self.device)
                model_names.append(model_name)

                self.attack_loader = AttackLoader(model)
                for attack_name in self.attack_loader.attacks_dict.keys():
                    logging.info(f"Running attack: {attack_name}")
                    attack = self.attack_loader.get_attack(attack_name)

                    adv_examples_list = []
                    for batch_idx, (data, labels) in enumerate(test_loader):
                        if batch_idx >= 3:  # Limit to first few batches
                            break
                        data, labels = data.to(self.device), labels.to(self.device)
                        adv_example = attack.attack(data, labels)

                        if len(adv_example) >= 2 and adv_example[1].size(0) > 0:
                            adv_examples_list.append(adv_example)

                    # Visualization and logging of adversarial examples
                    class_names = self.classes.get(dataset_name, [])
                    self.visualization.visualize_attack(adv_examples_list, model_names, self.current_task, dataset_name,
                                                        attack_name)
                    logging.info(
                        f"Generated {len(adv_examples_list)} adversarial examples for dataset: {dataset_name} "
                        f"using attack: {attack_name}")

                    all_results.append({
                        'dataset_name': dataset_name,
                        'model_name': model_name,
                        'attack_name': attack_name,
                        'adv_examples': adv_examples_list
                    })

        # Save all results into consolidated CSV file
        # self.save_results(all_results)




    def apply_defense(self, defense, defense_name, adv_examples_dict, dataset_name, model_name, all_results,
                      test_loader):
        defense_results = {}  # Store robustness results

        for attack_name, (adv_examples, adv_labels) in adv_examples_dict.items():
            defended_model, correct, total = defense.defend(torch.stack(adv_examples), torch.stack(adv_labels))

            # Train the defended model on adversarial examples if applicable
            adv_train_dataset = torch.utils.data.TensorDataset(torch.stack(adv_examples), torch.stack(adv_labels))
            defended_model, trainer = self.train_and_evaluate_model(model_name=model_name,
                                                                    dataset_name=dataset_name,
                                                                    train_loader=adv_train_dataset,
                                                                    val_loader=None,
                                                                    test_loader=test_loader,
                                                                    input_channels=self.input_channels_dict[
                                                                        dataset_name],
                                                                    adversarial=True)

            # Evaluate defended model
            true_labels, predictions = trainer.get_test_results()
            evaluator = Evaluator(defense_name, [], true_labels, predictions, self.current_task)
            evaluator.evaluate(dataset_name)
            all_results.append(evaluator)

            # Collect robustness results
            accuracy = (torch.tensor(predictions) == torch.tensor(true_labels)).float().mean().item()
            defense_results[defense_name] = {
                'attack_name': attack_name,
                'accuracy': accuracy,
                'num_examples': len(adv_examples),
            }

        return defense_results

    def compute_perturbation_data(self, defense, adv_examples_dict):
        perturbations = {}
        for attack_name, (adv_examples, _) in adv_examples_dict.items():
            perturbation_list = []
            for example in adv_examples:
                # Compute perturbation for each adversarial example
                perturbation = example - example.detach()
                perturbation_list.append(perturbation.abs().mean().item())
            perturbations[attack_name] = perturbation_list
        return perturbations

    def generate_adv_examples(self, test_loader):
        adv_examples_dict = {}
        for attack_name in self.attack_loader.attacks_dict.keys():
            logging.info(f"Running attack: {attack_name}")
            attack = self.attack_loader.get_attack(attack_name)

            adv_examples_list = []
            adv_labels_list = []
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                adv_example = attack.attack(data, labels)

                if adv_example:
                    adv_examples_list.append(adv_example[0])
                    adv_labels_list.append(adv_example[1])

            adv_examples_dict[attack_name] = (adv_examples_list, adv_labels_list)
        return adv_examples_dict

    def train_and_evaluate_model(self, model_name, dataset_name, train_loader, val_loader, test_loader, input_channels,
                                 depths, num_classes, class_names, adversarial=False):
        """Trains and evaluates models for each specified depth using DataLoader instances."""

        models = {}

        for depth in depths:
            try:
                model, model_name_with_depth = self.models_loader.get_model(
                    model_name, depth=depth, input_channels=input_channels, num_classes=num_classes
                )
                model.to(self.device)
            except ValueError as e:
                logging.error(f"Failed to load model {model_name} with depth {depth}: {e}")
                continue

            # Calculate and print the total number of parameters in the model
            total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
            logging.info(f'Total parameters for {model_name_with_depth}: {total_params:.2f}M')

            hyperparams = self.hyperparams_dict[dataset_name]
            optimizer = self.optimizers_dict.get_optimizer(
                hyperparams['optimizer'], model.parameters(),
                lr=hyperparams['lr'],
                momentum=hyperparams.get('momentum', 0)
            )

            if self.current_task in ['defense', 'attack']:
                adversarial = True

            if self.current_task is None or dataset_name is None:
                raise ValueError("Task name or dataset name is not set.")

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=self.criterion,
                model_name=model_name_with_depth,
                task_name=self.current_task,
                dataset_name=dataset_name,
                device=self.device,
                args=self.args,  # Pass args here
                attack_loader=self.attack_loader,
                scheduler=self.lr_scheduler_loader.get_scheduler(
                    hyperparams['scheduler'], optimizer, **hyperparams['scheduler_params']
                ) if self.lr_scheduler_loader else None,
                cross_validator=self.cross_validator
            )

            if adversarial:
                # Use adversarial training
                trainer.train(patience=hyperparams['patience'], adversarial=True)
            else:
                # Use normal training
                trainer.train(patience=hyperparams['patience'])

            trainer.test()
            model.trained = True

            models[depth] = (model, trainer)

        return models


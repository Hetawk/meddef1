# task_handler.py
import logging
import torch
from gan.attack.attack_loader import AttackLoader
from gan.defense.defense_loader import DefenseLoader
from train import Trainer
from utils.ensemble import Ensemble
from utils.evaluator import Evaluator
from utils.robustness.cross_validation import CrossValidator
from utils.visual.normal_visual import visualize_all
from utils.visual.visualization import Visualization


class TaskHandler:
    def __init__(self, datasets_dict, models_loader, optimizers_dict, hyperparams_dict, input_channels_dict, classes,
                 dataset_name, lr_scheduler_loader, cross_validator, device, args):
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

    def run_train(self):
        """Runs the training process for all datasets and models."""
        self.current_task = 'normal_training'
        logging.info("Training task selected.")
        all_results = []

        for dataset_name, (train_loader, val_loader, test_loader) in self.datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")
            input_channels = self.input_channels_dict.get(dataset_name)
            logging.info(f"Input channels for {dataset_name}: {input_channels}")

            model_list = []
            model_names_list = []
            true_labels_dict = {}
            predictions_dict = {}

            evaluator = None
            for model_name in self.models_loader.models_dict.keys():
                logging.info(f"Loading model: {model_name}")

                # Check if the model has already been trained
                if model_name in self.trained_models:
                    model, trainer = self.trained_models[model_name]
                    logging.info(f"Model {model_name} already trained. Skipping training.")
                else:
                    model, trainer = self.train_and_evaluate_model(model_name, dataset_name,
                                                                   train_loader, val_loader, test_loader,
                                                                   input_channels)
                    self.trained_models[model_name] = (model, trainer)

                model_list.append(model)
                model_names_list.append(model_name)
                true_labels, predictions = trainer.get_test_results()
                true_labels_dict[model_name] = [label.tolist() for label in true_labels]
                predictions_dict[model_name] = [pred.tolist() for pred in predictions]

                evaluator = Evaluator(model_name, [], true_labels, predictions, self.current_task)
                evaluator.evaluate(dataset_name)
                all_results.append(evaluator)

            # Extract unique labels from true_labels_dict
            unique_labels = set()
            for labels in true_labels_dict.values():
                unique_labels.update(labels)

            class_names = self.classes.get(dataset_name, [])
            # Ensure class_names includes all unique labels
            class_names = list(set(class_names).union(unique_labels))

            logging.info(f"True labels: {true_labels_dict}")
            logging.info(f"Predictions: {predictions_dict}")
            self.visualization.visualize_normal(model_names_list, (true_labels_dict, predictions_dict),
                                                self.current_task, dataset_name, class_names)
            if evaluator is not None:
                all_results.extend(evaluator.results)

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

    def run_defense(self):
        self.current_task = 'defense'
        logging.info("Defense task selected.")
        all_results = []

        selected_defenses = [
            'grad_mask',
            'randomization',
            'cert_defense',
            'def_distill',
            # Add more defenses as needed
        ]

        defenses = {}
        robustness_results = {}
        perturbations = {}
        adv_examples_dict = {}
        class_names = []

        for dataset_name, (train_loader, val_loader, test_loader) in self.datasets_dict.items():
            logging.info(f"Processing dataset: {dataset_name}")
            input_channels = self.input_channels_dict.get(dataset_name)
            # Fetch class names correctly
            class_names = self.classes.get(dataset_name, [])

            for model_name in self.models_loader.models_dict.keys():
                logging.info(f"Loading model: {model_name}")
                num_classes = len(class_names)  # Ensure num_classes is set correctly
                model = self.models_loader.load_pretrained_model(model_name, 'normal_training', dataset_name)
                model.to(self.device)

                # Generate adversarial examples
                self.attack_loader = AttackLoader(model)
                adv_examples_dict = self.generate_adv_examples(test_loader)

                # Apply selected defenses and collect results
                self.defense_loader = DefenseLoader(model)
                for defense_name in selected_defenses:
                    logging.info(f"Applying defense: {defense_name}")
                    defense = self.defense_loader.get_defense(defense_name)
                    defense_results = self.apply_defense(defense, defense_name, adv_examples_dict, dataset_name,
                                                         model_name, all_results, test_loader)

                    defenses[defense_name] = defense
                    robustness_results[defense_name] = defense_results
                    perturbations[defense_name] = self.compute_perturbation_data(defense, adv_examples_dict)

        # Call the visualization method with collected data
        if adv_examples_dict and class_names:
            self.visualization.visualize_defense(
                defenses=defenses,
                adv_examples_dict=adv_examples_dict,
                robustness_results=robustness_results,
                perturbations=perturbations,
                class_names=class_names,
                task_name=self.current_task,
                dataset_name=self.dataset_name
            )

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

    # utils/task_handler.py

    # utils/task_handler.py

    def train_and_evaluate_model(self, model_name, dataset_name, train_loader, val_loader, test_loader, input_channels,
                                 adversarial=False):
        """Trains and evaluates a model using DataLoader instances."""

        # Pass num_classes correctly
        class_names = self.classes.get(dataset_name, [])
        num_classes = len(class_names)
        if num_classes <= 2:
            raise ValueError(f"Number of classes for dataset {dataset_name} is {num_classes}, which seems incorrect.")
        model = self.models_loader.get_model(model_name, input_channels=input_channels, num_classes=num_classes)
        model.to(self.device)

        # Calculate and print the total number of parameters in the model
        total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
        logging.info(f'Total parameters for {model_name}: {total_params:.2f}M')

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
            model_name=model_name,
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

        return model, trainer

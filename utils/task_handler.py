import logging
import os
import gc
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from gan.defense.prune import Pruner
from train import Trainer

# from gan.defense.defense_loader import DefenseLoader
from loader.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from train import TrainingManager


class TaskHandler:
    def __init__(self, args):
        self.args = args
        self.training_manager = TrainingManager(args)
        # Add dataset_loader initialization
        self.dataset_loader = DatasetLoader()
        self.model_loader = ModelLoader(
            args.device, args.arch,
            getattr(args, 'pretrained', True),
            getattr(args, 'fp16', False)
        )

    def run_train(self):
       # "task_name normal_training"
        """Handle normal training workflow"""
        logging.info("Starting normal training task")
        for dataset_name in self.args.data:
            self.training_manager.train_dataset(dataset_name)

    def run_attack(self):
        """Generate adversarial examples for a dataset"""
        logging.info("Starting attack generation task")

        dataset_name = self.args.data[0]
        base_model_name = self.args.arch[0]  # e.g., meddef1_

        # Get depth from args.depth dictionary
        depth_dict = self.args.depth
        if not isinstance(depth_dict, dict):
            logging.error("Depth argument must be a dictionary")
            return

        # Get depths for the specified model
        depths = depth_dict.get(base_model_name, [])
        if not depths:
            logging.error(f"No depths specified for model {base_model_name}")
            return

        # Use first depth value since we're processing one model at a time
        depth = depths[0]

        # Format the full model name by combining arch and depth
        full_model_name = f"{base_model_name}_{depth}"
        logging.info(f"Processing model: {full_model_name}")

        # Load data for all splits at once
        train_loader, val_loader, test_loader = self.dataset_loader.load_data(
            dataset_name=dataset_name,
            batch_size={
                'train': self.args.train_batch,
                'val': self.args.train_batch,
                'test': self.args.train_batch
            },
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )

        # Get number of classes from the dataset: ensure that dataset has a 'classes' attribute.
        dataset = train_loader.dataset
        if hasattr(dataset, 'classes'):
            num_classes = len(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            num_classes = len(dataset.class_to_idx)
        else:
            raise AttributeError("Dataset does not contain class information.")

        # Initialize model using base_model_name and depth
        models_and_names = self.model_loader.get_model(
            model_name=base_model_name,  # Use base_model_name here
            depth=depth,                 # Use single depth value
            input_channels=3,
            num_classes=num_classes,
            task_name=self.args.task_name,
            dataset_name=dataset_name
        )

        if not models_and_names:
            logging.error("No models returned from model loader")
            return

        model, _ = models_and_names[0]  # Ignore returned model name
        model = model.to(self.args.device)
        model.eval()

        # Initialize attack components once
        from gan.defense.adv_train import AdversarialTraining
        attack_trainer = AdversarialTraining(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            config=self.args
        )

        attack_type = getattr(self.args, 'attack_type', 'fgsm')

        # Define attack percentages for each split
        attack_percentages = {
            'train': getattr(self.args, 'attack_train_percentage', 0.35),
            'val': 0.7,
            'test': 1.0
        }

        # Process each split using the same model
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

        for split, data_loader in data_loaders.items():
            logging.info(f"Generating attacks for {split} split")

            # Calculate the maximum number of samples to attack
            max_samples = int(len(data_loader.dataset) *
                              attack_percentages[split])
            total_batches = (
                max_samples + self.args.train_batch - 1) // self.args.train_batch

            # Update output directory to use full_model_name
            output_dir = os.path.join("out", "attacks", dataset_name,
                                      full_model_name, attack_type, split)
            os.makedirs(output_dir, exist_ok=True)

            try:
                # Initialize empty lists before the loop
                adversarial_images = []
                labels_list = []
                processed_samples = 0

                with tqdm(total=total_batches, desc=f"Generating attacks for {split}") as pbar:
                    for batch_idx, (data, target) in enumerate(data_loader):
                        if processed_samples >= max_samples:
                            break

                        try:
                            data = data.to(self.args.device)
                            target = target.to(self.args.device)

                            # Generate adversarial examples
                            _, adv_data, _ = attack_trainer.attack.attack(
                                data, target)

                            # Store results on CPU
                            adversarial_images.append(adv_data.cpu())
                            labels_list.append(target.cpu())

                            processed_samples += len(data)

                            # Memory management
                            del data, target, adv_data
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()  # Ensure CUDA operations are complete

                            pbar.update(1)
                            pbar.set_postfix({'samples': processed_samples})

                        except Exception as e:
                            logging.error(
                                f"Error in batch {batch_idx}: {str(e)}")
                            continue

                # Only process results if we have collected any
                if adversarial_images:
                    # Concatenate the results
                    adversarial_images = torch.cat(adversarial_images, dim=0)
                    labels_list = torch.cat(labels_list, dim=0)

                    # Save all results as .png images
                    output_dir_adv = os.path.join(output_dir, "adversarial")
                    os.makedirs(output_dir_adv, exist_ok=True)

                    for i in range(len(adversarial_images)):
                        save_image(adversarial_images[i], os.path.join(
                            output_dir_adv, f"adv_{split}_{i}.png"))

                    logging.info(
                        f"Saved {processed_samples} attacks as images to {output_dir_adv}")

                    # Save metadata
                    metadata = {
                        'total_samples': len(data_loader.dataset),
                        'attacked_samples': processed_samples,
                        'attack_percentage': attack_percentages[split],
                        'batch_size': self.args.train_batch,
                        'attack_type': attack_type,
                        'epsilon': self.args.epsilon,
                        'storage_format': 'png'
                    }
                    torch.save(metadata, os.path.join(
                        output_dir, "metadata.pt"))
                    logging.info(f"Saved metadata to {output_dir}/metadata.pt")
                else:
                    logging.warning(
                        f"No successful attacks generated for {split} split")

            except Exception as e:
                logging.error(
                    f"Error during attack generation for {split}: {str(e)}")
            finally:
                # Clean up iteration-specific memory
                for var in ['adversarial_images', 'labels_list']:
                    if var in locals():
                        del locals()[var]
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Final cleanup after all splits are processed
        del model, attack_trainer, models_and_names
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

    def run_defense(self):
        """Handle defense workflow: load, prune, and evaluate the model"""
        logging.info("Starting defense task: model pruning")
        dataset_name = self.args.data[0]
        base_model_name = self.args.arch[0]  # e.g., meddef1_
        depth_dict = self.args.depth
        if not isinstance(depth_dict, dict):
            logging.error("Depth argument must be a dictionary")
            return
        depths = depth_dict.get(base_model_name, [])
        if not depths:
            logging.error(f"No depths specified for model {base_model_name}")
            return
        depth = depths[0]
        full_model_name = f"{base_model_name}_{depth}"
        logging.info(f"Processing model for defense: {full_model_name}")

        # Load test data (or any split for evaluation)
        _, _, test_loader = self.dataset_loader.load_data(
            dataset_name=dataset_name,
            batch_size={'train': self.args.train_batch,
                        'val': self.args.train_batch,
                        'test': self.args.train_batch},
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )
        # Get number of classes from the dataset: ensure that dataset has a 'classes' attribute.
        dataset = test_loader.dataset
        if hasattr(dataset, 'classes'):
            num_classes = len(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            num_classes = len(dataset.class_to_idx)
        else:
            raise AttributeError("Dataset does not contain class information.")

        # Load the model using model_loader
        models_and_names = self.model_loader.get_model(
            model_name=base_model_name,
            depth=depth,
            input_channels=3,
            num_classes=num_classes,
            task_name=self.args.task_name,
            dataset_name=dataset_name
        )
        if not models_and_names:
            logging.error("No models returned from model loader")
            return
        model, _ = models_and_names[0]
        model = model.to(self.args.device)

        # Optionally load pretrained weights if provided; load directly via state_dict
        if hasattr(self.args, "model_path") and self.args.model_path:
            model.load_state_dict(torch.load(self.args.model_path))
            model.eval()
            logging.info(f"Loaded model weights from {self.args.model_path}")

        
        prune_rate = getattr(self.args, "prune_rate", 0.3)
        pruner = Pruner(model, prune_rate)
        pruned_model = pruner.unstructured_prune()
        logging.info("Model pruning completed.")

        # Instantiate a Trainer (which has a test() method) for evaluation
        
        trainer = Trainer(
            model=pruned_model,
            train_loader=test_loader,  # dummy loader for Trainer interface
            val_loader=test_loader,
            test_loader=test_loader,
            optimizer=None,
            criterion=torch.nn.CrossEntropyLoss(),
            model_name=full_model_name,
            task_name=self.args.task_name,
            dataset_name=dataset_name,
            device=self.args.device,
            config=self.args,
            scheduler=None
        )
        test_loss, test_accuracy = trainer.test()
        logging.info(
            f"Pruned Model evaluation: Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}")

        # Save the pruned model using the same pattern as training
        trainer.save_model(f"save_model/pruned_{full_model_name}.pth")
        logging.info(f"Pruned model saved as pruned_{full_model_name}.pth")

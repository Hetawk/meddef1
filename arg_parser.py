# import argparse
# import logging
# import os
# from loader.dataset_loader import DatasetLoader
# from loader.preprocess import Preprocessor
# from model.model_loader import ModelLoader
#
# class ArgParser:
#     def __init__(self):
#         self.parser = argparse.ArgumentParser(description='PyTorch Machine Learning Script')
#         self._add_arguments()
#         self.args = self.parser.parse_args()
#         self._validate_args()
#         self.dataset_loader = DatasetLoader(self.args.dataset_name, self.args.data_dir)
#         self.preprocessor = Preprocessor(model_type=self.args.model_type, dataset_name=self.args.dataset_name,
#                                          task_name=self.args.task, data_dir=self.args.data_dir,
#                                          hyperparams={'batch_size': self.args.batch_size})
#         self.model_loader = ModelLoader(self.args.device)
#
#     def _add_arguments(self):
#         self.parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
#         self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
#         self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
#         self.parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory for dataset')
#         self.parser.add_argument('--dataset_name', type=str, default='ccts', help='Name of the dataset to load')
#         self.parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
#         self.parser.add_argument('--model_type', type=str, default='resnet50', help='Type of model to use')
#         self.parser.add_argument('--num_classes', type=int, help='Number of output classes')
#         self.parser.add_argument('--task', type=str, default='normal_training', choices=['normal_training', 'attack', 'defense'], help='Task to run')
#
#     def _validate_args(self):
#         if self.args.epochs <= 0:
#             raise ValueError("Number of epochs must be positive")
#         if self.args.lr <= 0:
#             raise ValueError("Learning rate must be positive")
#         if self.args.batch_size <= 0:
#             raise ValueError("Batch size must be positive")
#         if not os.path.isdir(self.args.data_dir):
#             raise ValueError(f"Data directory {self.args.data_dir} does not exist")
#
#     def get_args(self):
#         return self.args
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     arg_parser = ArgParser()
#     args = arg_parser.get_args()
#     logging.info(f"Parsed arguments: {args}")
#
#     # Load datasets
#     train_dataset, val_dataset, test_dataset = arg_parser.dataset_loader.load()
#     logging.info(f"Loaded datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
#
#     # Preprocess datasets
#     train_dataset, val_dataset, test_dataset = arg_parser.preprocessor.preprocess(train_dataset, val_dataset, test_dataset, arg_parser.dataset_loader.get_input_channels())
#
#     # Log the size and shape of the preprocessed data
#     logging.info(f"Preprocessed train dataset size: {len(train_dataset)}")
#     logging.info(f"Preprocessed train dataset sample shape: {train_dataset[0][0].shape}")
#     if val_dataset:
#         logging.info(f"Preprocessed val dataset size: {len(val_dataset)}")
#         logging.info(f"Preprocessed val dataset sample shape: {val_dataset[0][0].shape}")
#     if test_dataset:
#         logging.info(f"Preprocessed test dataset size: {len(test_dataset)}")
#         logging.info(f"Preprocessed test dataset sample shape: {test_dataset[0][0].shape}")
#
#     # Wrap datasets in dataloaders
#     train_loader, val_loader, test_loader = arg_parser.preprocessor.wrap_datasets_in_dataloaders(train_dataset, val_dataset, test_dataset)
#     logging.info(f"Wrapped datasets in dataloaders: train_loader={len(train_loader)}, val_loader={len(val_loader) if val_loader else 0}, test_loader={len(test_loader) if test_loader else 0}")
#
#     # Visualize samples
#     arg_parser.preprocessor.visualize_samples('default_model', train_dataset)
#
#     # Print class counts
#     arg_parser.dataset_loader.print_class_counts()
#
#     # Get number of classes dynamically if not provided
#     if args.num_classes is None:
#         classes = arg_parser.preprocessor.extract_classes(train_dataset)
#         num_classes = len(classes)
#     else:
#         num_classes = args.num_classes
#
#     # Get model
#     model = arg_parser.model_loader.get_model(args.model_type, input_channels=arg_parser.dataset_loader.get_input_channels(), num_classes=num_classes)
#     logging.info(f"Model {args.model_type} loaded with {arg_parser.dataset_loader.get_input_channels()} input channels and {num_classes} output classes.")
#
#     # Run task
#     if args.task == 'normal_training':
#         logging.info("Running normal training task.")
#         # Implement normal training task
#     elif args.task == 'attack':
#         logging.info("Running attack task.")
#         # Implement attack task
#     elif args.task == 'defense':
#         logging.info("Running defense task.")
#         # Implement defense task
#     else:
#         logging.error(f"Unknown task: {args.task}. No task was executed.")

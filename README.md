# MedDef

MedDef is a machine learning project designed to modularize model training in a scalable way, with a particular focus on adversarial resilience in medical imaging. The project aims to provide robust defense mechanisms against adversarial attacks in medical image analysis, ensuring the reliability and accuracy of machine learning models in critical healthcare applications.

## Features

- Modularized model training
- Support for various datasets and model architectures
- Adversarial training and defense mechanisms
- Cross-validation and hyperparameter tuning
- Logging and visualization of training and evaluation metrics

## Installation

To get started with MedDef, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hetawk/meddef.git
cd meddef
pip install -r requirements.txt
```

## Usage

To run the main script, use the following command:

```bash
python main.py --data chest_xray --task_name normal_training --epochs 100 --train-batch 32 --test-batch 32 --lr 0.001 --drop 0.5 --gpu-ids 2 --arch resnet --depth '{"resnet": [18, 34]}' --pin_memory
```

### Command Line Arguments

- `--data`: The dataset to use (e.g., `chest_xray`)
- `--task_name`: The task to perform (`normal_training`, `attack`, `defense`)
- `--epochs`: Number of training epochs
- `--train-batch`: Batch size for training
- `--test-batch`: Batch size for testing
- `--lr`: Learning rate
- `--drop`: Dropout rate
- `--gpu-ids`: GPU IDs to use
- `--arch`: Model architecture (e.g., `resnet`)
- `--depth`: Depth of the model architecture (e.g., `{"resnet": [18, 34]}`)
- `--pin_memory`: Use pinned memory for data loading

## Project Structure

- `main.py`: The main script to run the project
- `loader/`: Contains dataset loading utilities
- `model/`: Contains model definitions and loading utilities
- `utils/`: Contains utility functions for logging, optimization, and task handling
- `arg_parser.py`: Argument parser for command line arguments

## Running chect_xray

Here is an example command to run the project with the `chest_xray` dataset and `resnet` architecture:

```bash
python main.py --data chest_xray --task_name normal_training --epochs 100 --train-batch 32 --test-batch 32 --lr 0.001 --drop 0.5 --gpu-ids 2 --arch resnet --depth '{"resnet": [18, 34]}' --pin_memory
```

## When running with config.yaml configuration overriding arg_parser

```bash
python main.py --data ccts --data_key ccts --task_name normal_training --epochs 2 --train-batch 8 --test-batch 8 --lr 0.001 --drop 0.5 --gpu-ids 0 --arch [resnet, meddef1_] --depth "{'resnet': [18,34], 'meddef1_': [1.0,1.1]}"
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

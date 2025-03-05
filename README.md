# MedDef

MedDef is a machine learning project designed to modularize model training in a scalable way, with a particular focus on adversarial resilience in medical imaging. The project aims to provide robust defense mechanisms against adversarial attacks in medical image analysis, ensuring the reliability and accuracy of machine learning models in critical healthcare applications.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command Arguments](#basic-command-arguments)
  - [Dataset Organization](#dataset-organization)
  - [Dataset Processing](#dataset-processing)
  - [Model Training](#model-training)
  - [Adversarial Training](#adversarial-training)
  - [Testing Models](#testing-models)
  - [Defense Mechanisms](#defense-mechanisms)
  - [Robustness Evaluation](#robustness-evaluation)
  - [Visualization Tools](#visualization-tools)
- [Output Directory Structure](#output-directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Modularized model training architecture
- Support for various datasets (chest_xray, ccts, rotc, etc...)
- Multiple model architectures (ResNet, DenseNet, VGG, custom MedDef models)
- Adversarial attack generation and defense evaluation
- Model pruning for improved robustness
- Comprehensive evaluation tools
- Visualization tools for model interpretability (saliency maps)
- Cross-validation and hyperparameter tuning
- Logging and visualization of training and evaluation metrics

## Project Structure

```
meddef1/
├── main.py                 # Main script for training, attack and defense
├── test.py                 # Model testing script
├── train.py                # Core training functionality
├── dataset_processing.py   # Dataset preparation utilities
├── evaluate_attacks.py     # Script for evaluating model against attacks
├── loader/                 # Dataset loading and processing modules
├── model/                  # Model architectures and definitions
├── utils/                  # Utility functions for various tasks
├── gan/                    # GAN-related implementations
├── run/                    # Run scripts and configurations
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

To get started with MedDef, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hetawk/meddef1.git
cd meddef1
pip install -r requirements.txt
```

## Usage

### Basic Command Arguments

Common arguments used across different commands:

- `--data`: Dataset name (e.g., chest_xray, rotc, ccts)
- `--arch`: Model architecture (e.g., resnet, densenet, vgg, meddef1)
- `--depth`: Architecture depth/version (e.g., '{"resnet": [18, 34]}')
- `--task_name`: Task to perform (normal_training, attack, defense)
- `--epochs`: Number of training epochs
- `--train_batch`: Batch size for training
- `--lr`: Learning rate
- `--drop`: Dropout rate
- `--gpu-ids`: GPU IDs to use
- `--optimizer`: Optimizer to use (adam, sgd)
- `--pin_memory`: Use pinned memory for data loading

### Dataset Organization

The project uses a specific directory structure for datasets:

1. **Original Data**: Place your raw, unprocessed datasets in the `dataset/` directory.

   ```
   dataset/
   ├── chest_xray/
   ├── rotc/
   ├── ccts/
   └── your_new_dataset/
   ```

2. **Dataset Naming**: The project recognizes datasets by short names, which are:

   ```
   'ccts', 'rotc', 'chest_xray',
   ```

   When adding your own dataset, either use one of these names or add your dataset name to the `SUPPORTED_DATASETS` list in `loader/dataset_loader.py` and all data related usage areas.

3. **Processed Data**: After processing, datasets are stored in the `processed_data/` directory:

   ```
   processed_data/
   ├── chest_xray/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── rotc/
   └── your_processed_dataset/
   ```

If you store your data in a different location, you'll need to modify the paths in the code accordingly.

### Dataset Processing

Before training, you may need to process your datasets:

```bash
python dataset_processing.py --data ccts --output_dir processed_data
```

### Model Training

#### Basic Training Examples


```bash
# Train with ResNet on chest_xray dataset
python main.py --data chest_xray --task_name normal_training --epochs 100 --train_batch 32 \
  --test-batch 32 --lr 0.001 --drop 0.3 --gpu-ids 0 --arch resnet --depth '{"resnet": [18]}' --pin_memory

# Train with custom MedDef architecture
python main.py --data ccts --arch meddef1 --depth '{"meddef1": [1.0, 1.1, 1.2]}' \
  --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam

# Train with DenseNet on ROTC dataset
python main.py --data rotc --arch densenet --depth '{"densenet": [121]}' \
  --train_batch 64 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
```

#### Advanced Training Options

```bash
# Training with advanced learning rate scheduling
  python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0]}' \
  --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.3 --num_workers 4 \
  --pin_memory  --lr-scheduler cosine --gpu-ids 0 --task_name normal_training --optimizer adam \
  --patience 15 --weight-decay 1e-6 --lambda_l2 0.0001


  python main.py --data chest_xray --arch resnet --depth '{"resnet": [18,34,50,101,152]}' \
  --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.3 --num_workers 4 \
  --pin_memory  --lr-scheduler cosine --gpu-ids 0 --task_name normal_training --optimizer adam \
  --patience 15 --weight-decay 1e-6 --lambda_l2 0.0001
```

### Adversarial Training

#### Two-Step Attack Generation and Training

```bash
# Step 1: Generate and save attacks if you wish to use pre-generated attacks for faster training or robustly generate attaack on the fly; which means you don't have to generate attacks but you can specify the adversarial argument and attacks will be generate on the fly while training the model 

python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.0]}' \
  --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

# Step 2: Train with pre-generated attacks
python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.0]}' \
  --train_batch 32 --epochs 5 --lr 0.001 --drop 0.3 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam \
  --adversarial --attack_type fgsm
```

#### Basic Adversarial Training

```bash
# Train with single attack type (FGSM)
python main.py --data chest_xray --arch meddef1_ --depth '{"meddef1_": [1.0]}' \
  --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam \
  --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm
```

#### Multi-Attack Adversarial Training

```bash
# Train with multiple attack types
python main.py --data chest_xray --arch meddef1_ --depth '{"meddef1_": [1.0, 1.1, 1.2]}' \
  --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam \
  --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm bim jsma pgd
```

### Testing Models

#### Test on a Single Image

```bash
# Test on a normal image
python test.py --data rotc --arch meddef1_ --depth 1.0 \
  --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" \
  --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"
```

#### Test on Adversarial Images

```bash
# Test on adversarial image
python test.py --data rotc --arch meddef1 --depth 1.0 \
  --model_path "out/normal_training/rotc/meddef1_1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" \
  --image_path "out/normal_training/rotc/meddef1_1.0/attack/bim+jsma/sample_0_adv.png"
# Test on original image from the attack dir
python test.py --data rotc --arch meddef1 --depth 1.0 \
  --model_path "out/normal_training/rotc/meddef1_1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" \
  --image_path "out/normal_training/rotc/meddef1_1.0/attack/bim+jsma/sample_0_orig.png"

# Test on original image from the preprocessed_data dir
python test.py --data rotc --arch meddef1 --depth 1.0 \
  --model_path "out/normal_training/rotc/meddef1_1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" \
  --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"

```

### Defense Mechanisms

#### Model Pruning as a Defense

```bash
# Apply pruning to a trained model
python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.0]}' \
  --task_name defense \
  --model_path "out/normal_training/rotc/meddef1_1.0/save_model/best_meddef1_1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" \
  --prune_rate 0.3

# Test a pruned model
python test.py --data rotc --arch meddef1 --depth 1.0 \
  --model_path "out/defense/rotc/meddef1_1.0/save_model/pruned_meddef1__1.0_epochs100_lr0.001_batch32_20250224.pth" \
  --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg" \
  --task_name defense

python test.py --data rotc --arch meddef1 --depth 1.0 \
  --model_path "out/defense/rotc/meddef1_1.0/save_model/pruned_meddef1__1.0_epochs100_lr0.001_batch32_20250224.pth" \
  --image_path "out/normal_training/rotc/meddef1_1.0/attack/pgd/sample_0_adv.png" \
  --task_name defense
```

### Robustness Evaluation

#### Comprehensive Model Evaluation

```bash
# Evaluate model against multiple attacks and pruning rates
python evaluate_attacks.py --data chest_xray --arch meddef1 --depth 1.0 \
  --model_path "out/normal_training/chest_xray/meddef1_1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" \
  --attack_types pgd fgsm --attack_eps 0.2 \
  --prune_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 0
```

#### Comparative Model Analysis

```bash
# Compare models across different architectures
python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types fgsm pgd bim jsma --attack_eps 0.2 --prune_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 0
```

### Visualization Tools

#### Generating Saliency Maps

```bash
# Generate saliency map for a single image
python -m loader.saliency_generator --data rotc --arch densenet --depth 121 \
  --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" \
  --image_path "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_adv.png"

# Process multiple images for saliency maps
python -m loader.saliency_generator --data rotc --arch densenet --depth 121 \
  --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" \
  --image_paths "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_orig.png" \
                "out/normal_training/rotc/densenet_121/attack/fgsm/sample_3_orig.png" \
                "out/normal_training/rotc/densenet_121/attack/fgsm/sample_4_orig.png"
```

## Output Directory Structure

MedDef automatically organizes all outputs (models, visualizations, attack examples, etc.) in a hierarchical directory structure:

```
out/
├── normal_training/           # Regular training results
│   ├── chest_xray/            # Organized by dataset
│   │   ├── resnet_18/         # Organized by model architecture and depth
│   │   │   ├── save_model/    # Model checkpoints
│   │   │   ├── attack/        # Generated attacks
│   │   │   ├── visualization/ # Visualizations
│   │   │   └── all_evaluation_metrics.csv
│   │   └── meddef1_1.0/
│   └── rotc/
│       ├── densenet_121/
│       ├── meddef1_1.0/
│       │   ├── adv/           # Adversarial training results
│       │   ├── attack/        # Generated attacks (e.g. fgsm, pgd)
│       │   ├── save_model/    # Model checkpoints
│       │   ├── visualization/ # Visualizations
│       │   └── all_evaluation_metrics.csv
│       ├── resnet_18/
│       └── vgg_16/
├── defense/                   # Defense mechanism results
├── attacks/                   # Stored adversarial examples
└── attack_evaluation/         # Attack evaluation results
```

Model checkpoints follow a consistent naming pattern:

```
best_[model_name]_[dataset]_epochs[num_epochs]_lr[learning_rate]_batch[batch_size]_[date].pth
```

For example: `best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250221.pth`

This structured organization makes it easy to locate specific results and compare performance across different architectures, datasets, and training methods.

## Star the Repository

If you find MedDef useful for your research or projects, please consider giving it a star on GitHub. This helps others discover the project and motivates future development.

[![GitHub stars](https://img.shields.io/github/stars/hetawk/meddef1.svg?style=social&label=Star)](https://github.com/hetawk/meddef1)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

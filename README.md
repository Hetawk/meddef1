# MedDef

MedDef is a machine learning project designed to modularize model training in a scalable way, with a particular focus on adversarial resilience in medical imaging. The project aims to provide robust defense mechanisms against adversarial attacks in medical image analysis, ensuring the reliability and accuracy of machine learning models in critical healthcare applications.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command Arguments](#basic-command-arguments)
  - [Dataset Processing](#dataset-processing)
  - [Model Training](#model-training)
  - [Adversarial Training](#adversarial-training)
  - [Testing Models](#testing-models)
  - [Defense Mechanisms](#defense-mechanisms)
  - [Robustness Evaluation](#robustness-evaluation)
  - [Visualization Tools](#visualization-tools)
- [Contributing](#contributing)
- [License](#license)

## Features

- Modularized model training architecture
- Support for various datasets (chest_xray, CCTS, ROTC)
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
- `--arch`: Model architecture (e.g., resnet, densenet, vgg, meddef1\_)
- `--depth`: Architecture depth/version (e.g., '{"resnet": [18, 34]}')
- `--task_name`: Task to perform (normal_training, attack, defense)
- `--epochs`: Number of training epochs
- `--train_batch`: Batch size for training
- `--test-batch`: Batch size for testing
- `--lr`: Learning rate
- `--drop`: Dropout rate
- `--gpu-ids`: GPU IDs to use
- `--optimizer`: Optimizer to use (adam, sgd)
- `--pin_memory`: Use pinned memory for data loading

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
  --test-batch 32 --lr 0.001 --drop 0.5 --gpu-ids 0 --arch resnet --depth '{"resnet": [18]}' --pin_memory

# Train with custom MedDef architecture
python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0, 1.1, 1.2]}' \
  --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam

# Train with DenseNet on ROTC dataset
python main.py --data rotc --arch densenet --depth '{"densenet": [121]}' \
  --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
```

#### Advanced Training Options

```bash
# Training with advanced learning rate scheduling
python main.py --data ccts --data_key ccts --task_name normal_training --epochs 100 \
  --train_batch 32 --test-batch 32 --lr 0.01 --min-lr 1e-7 --warmup-epochs 10 \
  --lr-scheduler cosine --weight-decay 1e-6 --lambda_l2 0.0001 --drop 0.2 \
  --gpu-ids 0 --arch meddef2_ --depth "{'meddef2_': [2.0, 2.1]}" \
  --adv_training --max-grad-norm 1.0 --accumulation_steps 4 --patience 15
```

### Adversarial Training

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

#### Two-Step Attack Generation and Training

```bash
# Step 1: Generate and save attacks
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' \
  --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

# Step 2: Train with pre-generated attacks
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' \
  --train_batch 32 --epochs 5 --lr 0.001 --drop 0.3 --num_workers 4 \
  --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam \
  --adversarial --attack_type fgsm
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
python test.py --data rotc --arch meddef1_ --depth 1.0 \
  --model_path "out/normal_training/rotc/b_meddef1__1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" \
  --image_path "out/normal_training/rotc/meddef1__1.0/attack/bim+jsma/sample_0_orig.png"
```

### Defense Mechanisms

#### Model Pruning as a Defense

```bash
# Apply pruning to a trained model
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' \
  --task_name defense \
  --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" \
  --prune_rate 0.3

# Test a pruned model
python test.py --data rotc --arch meddef1_ --depth 1.0 \
  --model_path "out/defense/rotc/meddef1__1.0/save_model/pruned_meddef1__1.0_epochs100_lr0.001_batch32_20250224.pth" \
  --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg" \
  --task_name defense
```

### Robustness Evaluation

#### Comprehensive Model Evaluation

```bash
# Evaluate model against multiple attacks and pruning rates
python evaluate_attacks.py --data chest_xray --arch meddef1_ --depth 1.0 \
  --model_path "out/normal_training/chest_xray/meddef1__1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" \
  --attack_types fgsm pgd bim --attack_eps 0.1 \
  --prune_rates 0.0 0.1 0.3 0.5 0.7 \
  --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 0
```

#### Comparative Model Analysis

```bash
# Compare models across different architectures
python evaluate_attacks.py --data rotc --arch densenet --depth 121 \
  --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" \
  --attack_types fgsm pgd bim jsma --attack_eps 0.2 \
  --prune_rates 0.0 0.1 0.2 0.3 0.4 0.5 \
  --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 0
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

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

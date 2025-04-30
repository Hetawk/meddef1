# MedDef: An Efficient Self-Attention Model for Adversarial Resilience in Medical Imaging with Unstructured Pruning

MedDef is a machine learning project designed to modularize model training in a scalable way, with a particular focus on adversarial resilience in medical imaging. The project aims to provide robust defense mechanisms against adversarial attacks in medical image analysis, ensuring the reliability and accuracy of machine learning models in critical healthcare applications.

## Model Architecture

The MedDef architecture incorporates a Defense-Aware Attention Mechanism (DAAM) with strategic unstructured pruning to achieve robust adversarial resilience.

![MedDef Architecture](meddef.jpg)

## Features

- Modularized model training
- Support for various datasets and model architectures
- Adversarial training and defense mechanisms
- Cross-validation and hyperparameter tuning
- Logging and visualization of training and evaluation metrics

## Installation

To get started with MedDef, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hetawk/meddef1.git
cd meddef1
pip install -r requirements.txt
```

## Usage

### Basic Command Structure

```bash
python main.py --data <dataset> --task_name <task> --epochs <num> --train_batch <size> --test-batch <size> --lr <rate> --drop <rate> --gpu-ids <id> --arch <architecture> --depth <depth_config> [options]
```

### Command Line Arguments

- `--data`: The dataset to use (e.g., `chest_xray`, `rotc`, `ccts`)
- `--task_name`: The task to perform (`normal_training`, `attack`, `defense`)
- `--epochs`: Number of training epochs
- `--train_batch`: Batch size for training
- `--test-batch`: Batch size for testing
- `--lr`: Learning rate
- `--drop`: Dropout rate
- `--gpu-ids`: GPU IDs to use for training (e.g., `0`, `1`, `2`, `3`)
- `--arch`: Model architecture (e.g., `resnet`, `meddef1`, `densenet`, `vgg`)
- `--depth`: Depth of the model architecture (e.g., `{"resnet": [18, 34]}`)
- `--pin_memory`: Use pinned memory for data loading
- `--optimizer`: Optimizer to use (e.g., `adam`, `sgd`)
- `--weight_decay`: Weight decay for regularization
- `--adversarial`: Enable adversarial training
- `--attack_type`: Type of adversarial attack (e.g., `fgsm`, `pgd`, `bim`)
- `--attack_eps`: Epsilon value for adversarial attacks
- `--adv_weight`: Weight for adversarial loss
- `--enforce_split`: Enforce custom train/val/test splits. Useful in cases where there's imbalance in the dataset.
- `--train_split`: Proportion of training data
- `--val_split`: Proportion of validation data
- `--test_split`: Proportion of test data
- `--verify_classes`: Verify classes in the dataset, it helps ensure that the dataset is correctly structured and contains the expected classes. Note that passing this argument will not process the dataset. Instead it will only verify the classes.
- `--num_workers`: Number of workers for data loading
-

## Project Structure

- `main.py`: The main script to run the project
- `loader/`: Contains dataset loading utilities
- `model/`: Contains model definitions and loading utilities
- `utils/`: Contains utility functions for logging, optimization, and task handling
- `argument_parser.py`: Argument parser for command line arguments
- `test.py`: Script for testing trained models
- `evaluate_attacks.py`: Script for evaluating model robustness against attacks
- `dataset_processing.py`: Script for processing datasets
- `out/`: A dedicated dir where all outputs like visualization, model checkpoint, csv and txt files are save.
- `processed_data/`: A dedicated dir where all processed data is saved.
- `out/attack_evaluation`: A directory where the results of the attack evaluation are saved. This includes the results of the adversarial attacks on the model, such as Model Accuracy, Attack Success Rate (ASR) under different prunning condition and more.
- `out/saliency_maps`: A directory where the saliency maps generated for the images are saved. Saliency maps are visual representations of the regions in an image that are most important for the model's predictions.
- `out/runs`: A directory where the TensorBoard logs are saved. This includes the training and validation metrics, which can be visualized using TensorBoard.
- `out/normal_training`: A dir where both normal and adversarial training results are saved. This includes the model checkpoints, training logs, and other relevant files.
- `out/defense`: A dir where the results of the defense mechanism are saved when the `--task_name defense` is passed. This includes the pruned model checkpoints, and other relevant files.

## Examples

### Data Processing

```bash
# Process dataset using default settings
python dataset_processing.py --datasets chest_xray
# Process dataset with custom splits (80% train, 10% val, and 10% test)
python dataset_processing.py --datasets chest_xray --enforce_split --train_split 0.8 --val_split 0.1 --test_split 0.1

# Process dataset with custom splits (70% train, 15% val and 15% test)
python dataset_processing.py --datasets chest_xray --enforce_split --train_split 0.70 --val_split 0.15 --test_split 0.15
# Verify classes in a dataset
python dataset_processing.py --datasets rotc --verify_classes
```

### Standard Training

```bash
# MedDef model
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam

# DenseNet model
python main.py --data chest_xray --arch densenet --depth '{"densenet": [121]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam
```

### Adversarial Training

```bash
# MedDef with adversarial training
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --gpu-ids 0 --pin_memory --weight_decay 1e-4 --adversarial --attack_eps 0.1 --adv_weight 0.5 --attack_type pgd --task_name normal_training --optimizer adam

# MedDef with adversarial training with different lr
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0]}' --train_batch 32 --epochs 100 --lr 0.00005 --drop 0.5 --gpu-ids 0 --pin_memory --weight_decay 1e-4 --adversarial --attack_eps 0.1 --adv_weight 0.5 --attack_type pgd --task_name normal_training --optimizer adam

# DenseNet with adversarial training
python main.py --data chest_xray --arch densenet --depth '{"densenet": [121]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm
```

### Testing Models

```bash
# Test MedDef model
python test.py --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/adv/save_model/best_meddef1_1.0_chest_xray_epochs100_lr5e-05_batch32_20250402.pth"

# Test DenseNet model
python test.py --data chest_xray --arch densenet --depth 121 --model_path "out/normal_training/chest_xray/densenet_121/adv/save_model/best_densenet_121_chest_xray_epochs100_lr5e-05_batch32_20250331.pth"

# Performing an image test for MedDef
python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/defense/rotc/meddef1_1.0/save_model/pruned_meddef1_1.0_epochs100_lr0.001_batch32_20250224.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg" --task_name defense
```

### Evaluating Robustness

```bash
# Evaluate MedDef against multiple attacks and pruning rates
python evaluate_attacks.py --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/adv/save_model/best_meddef1_1.0_chest_xray_epochs100_lr5e-05_batch32_20250402.pth" --attack_types fgsm pgd bim jsma --attack_eps 0.05 --prune_rates 0.1 0.3 0.5 0.7 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1

# Evaluate DenseNet against multiple attacks and pruning rates
python evaluate_attacks.py --data chest_xray --arch densenet --depth 121 --model_path "out/normal_training/chest_xray/densenet_121/adv/save_model/best_densenet_121_chest_xray_epochs100_lr5e-05_batch32_20250331.pth" --attack_types fgsm pgd bim jsma --attack_eps 0.05 --prune_rates 0.1 0.3 0.5 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1
```

### Generating Saliency Maps

```bash
# Generate saliency maps for MedDef
python -m loader.saliency_generator --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/adv/save_model/best_meddef1_1.0_chest_xray_epochs100_lr5e-05_batch32_20250402.pth" --image_path "out/normal_training/chest_xray/meddef1_1.0/attack/pgd/sample_0_orig.png"

# Generate saliency maps for MedDef with 3 images
python -m loader.saliency_generator --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/adv/save_model/best_meddef1_1.0_chest_xray_epochs100_lr5e-05_batch32_20250402.pth"  --image_path "out/normal_training/chest_xray/resnet_18/attack/pgd/sample_4_orig.png" "out/normal_training/chest_xray/meddef1_1.0/attack/pgd/sample_3_orig.png" "out/normal_training/chest_xray/meddef1_1.0/attack/pgd/sample_0_orig.png"
# Generate saliency maps for DenseNet
python -m loader.saliency_generator --data chest_xray --arch densenet --depth 121 --model_path "out/normal_training/chest_xray/densenet_121/adv/save_model/best_densenet_121_chest_xray_epochs100_lr5e-05_batch32_20250331.pth" --image_path "out/normal_training/chest_xray/densenet_121/attack/pgd/sample_0_orig.png"
```

### Visualizing Results

```bash
# Launch TensorBoard to visualize training metrics
tensorboard --logdir=out/runs
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Support

If you find this project useful, please consider giving it a star on GitHub! Your support helps us improve and maintain this project.

[![GitHub stars](https://img.shields.io/github/stars/hetawk/meddef1.svg?style=social&label=Star)](https://github.com/hetawk/meddef1)

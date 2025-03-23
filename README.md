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
python main.py --data chest_xray --task_name normal_training --epochs 100 --train_batch 32 --test-batch 32 --lr 0.001 --drop 0.5 --gpu-ids 2 --arch resnet --depth '{"resnet": [18, 34]}' --pin_memory



```

### Command Line Arguments

- `--data`: The dataset to use (e.g., `chest_xray`)
- `--task_name`: The task to perform (`normal_training`, `attack`, `defense`)
- `--epochs`: Number of training epochs
- `--train_batch`: Batch size for training
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
python main.py --data chest_xray --task_name normal_training --epochs 100 --train_batch 32 --test-batch 32 --lr 0.001 --drop 0.5 --gpu-ids 2 --arch resnet --depth '{"resnet": [18, 34]}' --pin_memory
```


```bash
python dataset_processing.py --data ccts --output_dir processed_data

python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0]}' --train_batch 32 --epochs 3 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam




python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.2]}' --train_batch 64 --epochs 100 --lr 0.0001 --drop 0.3 --weight_decay 0.0001 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --attack_eps 0.2 --adv_weight 0.3 --attack_type pgd

## 

python main.py --data rotc --arch resnext --depth '{"resnext": [50]}' --train_batch 64 --epochs 2 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam


### test

python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/meddef1_1.0/save_model/best_meddef1_1.0_rotc_epochs100_lr0.001_batch32_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"


python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/meddef1_1.0/save_model/best_meddef1_1.0_rotc_epochs100_lr0.001_batch32_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"

python test.py --data ccts --arch meddef1 --depth '{"meddef1": [1.0]}' --test_batch 32 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --model_path "out/normal_training/ccts/meddef1_1.0/save_model/best_meddef1_1.0_ccts_epochs5_lr0.001_batch32_20250217.pth"


```


Adversarial Traiing
```bash
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0, 1.1,1.2]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm bim jsma pgd


python main.py --data chest_xray --arch vgg --depth '{"vgg": [16]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm

python main.py --data chest_xray --arch densenet --depth '{"densenet": [121]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm

python main.py --data chest_xray --arch resnet --depth '{"resnet": [18,34]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm

```


```bash
# Step 1: Generate and save attacks
python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.0]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

# Step 2: Train with pre-generated attacks
python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.0]}' --train_batch 32 --epochs 5 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm


```


### defense
```bash
### test before prune -> normal image normal train
python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/meddef1_1.0/adv/save_model/best_meddef1_1.0_rotc_epochs100_lr0.001_batch64_20250224.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"

### test before prune -> adversarial image normal train
## 1 | Confidence  1.0000
python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/b_meddef1_1.0/best_meddef1_1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" --image_path "out/normal_training/rotc/meddef1_1.0/attack/bim+jsma/sample_0_orig.png"
## 2 | Confidence 0.9780
python test.py --data rotc --arch resnet --depth 18 --model_path "out/normal_training/rotc/resnet_18/adv/save_model/best_resnet_18_rotc_epochs100_lr0.001_batch32_20250227.pth" --image_path "out/normal_training/rotc/resnet_18/attack/fgsm/sample_0_adv.png"
## 3 | Confidence 0.9154
python test.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --image_path "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_adv.png"
## 4 | Confidence 0.9999
python test.py --data rotc --arch vgg --depth 16 --model_path "out/normal_training/rotc/vgg_16/adv/save_model/best_vgg_16_rotc_epochs100_lr0.001_batch64_20250224.pth" --image_path "out/normal_training/rotc/vgg_16/attack/fgsm/sample_0_adv.png"

##### Chest Xray
## 1 | Confidence  1.0000
python test.py --data chest_xray --arch vgg --depth 16 --model_path "out/normal_training/chest_xray/vgg_16/adv/save_model/best_vgg_16_chest_xray_epochs100_lr0.0001_batch32_20250303.pth" --image_path "out/normal_training/chest_xray/vgg_16/attack/fgsm/sample_0_adv.png"

## 2 | Confidence 
python test.py --data chest_xray --arch resnet --depth 18 --model_path "out/normal_training/chest_xray/resnet_18/adv/save_model/best_resnet_18_chest_xray_epochs100_lr0.0005_batch32_20250227.pth" --image_path "out/normal_training/chest_xray/resnet_18/attack/fgsm/sample_0_adv.png"

## 3 | Confidence
python test.py --data chest_xray --arch densenet --depth 121 --model_path "out/normal_training/chest_xray/densenet_121/adv/save_model/best_densenet_121_chest_xray_epochs100_lr0.0001_batch32_20250303.pth" --image_path "out/normal_training/chest_xray/densenet_121/attack/fgsm/sample_0_adv.png"


## 4 | Confidence 
python test.py --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/adv/save_model/best_meddef1_1.0_chest_xray_epochs100_lr0.0001_batch32_20250303.pth" --image_path "out/normal_training/chest_xray/meddef1_1.0/attack/fgsm/sample_0_adv.png"

```


```bash

### test before prune -> normal image | adversarial train


## pruning
python main.py --data rotc --arch meddef1 --depth '{"meddef1": [1.0]}' --task_name defense --model_path "out/normal_training/rotc/meddef1_1.0/save_model/best_meddef1_1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" --prune_rate 0.3


### test before prune
python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/meddef1_1.0/save_model/best_meddef1_1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"


### test adversarial
python test.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/defense/rotc/meddef1_1.0/save_model/pruned_meddef1_1.0_epochs100_lr0.001_batch32_20250224.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg" --task_name defense


```

### Robustness Test
```bash
#### Evaluate a single model against multiple attacks and pruning rates:
python evaluate_attacks.py --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/save_model/best_meddef1_1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" --attack_types fgsm pgd bim --attack_eps 0.1 --prune_rates 0.1 0.3 0.5 0.7 --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 1

#### Compare a specific attack at a single pruning rate:
python evaluate_attacks.py --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/save_model/best_meddef1_1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" --attack_types fgsm --attack_eps 0.1 --prune_rates 0.3 --gpu-ids 1

#### testing different models

# For ResNet model
python evaluate_attacks.py --data chest_xray --arch resnet  --depth 18 --model_path "out/normal_training/chest_xray/resnet_18/save_model/best_resnet_18_chest_xray_epochs100_lr0.001_batch32_20250227.pth" --attack_types fgsm pgd --prune_rates 0.3 0.5 --gpu-ids 1

# For MedDef model
python evaluate_attacks.py --data chest_xray --arch meddef1 --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1_1.0/save_model/best_meddef1_1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" --attack_types fgsm pgd --prune_rates 0.3 0.5 --gpu-ids 1

# For Densenet
python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types fgsm pgd bim --attack_eps 0.2 --prune_rates 0.1 0.3 0.5 0.7 --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 1

python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types cw zoo boundary elasticnet onepixel fgsm pgd bim jsma --attack_eps 0.2 --prune_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 1


python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types fgsm --attack_eps 0.2 --prune_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1


python evaluate_attacks.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/b_meddef1_1.0/best_meddef1_1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" --attack_types fgsm --attack_eps 0.2 --prune_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1

python evaluate_attacks.py --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/b_meddef1_1.0/best_meddef1_1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" --attack_types fgsm pgd bim jsma --attack_eps 0.2 --prune_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1

```


### Salient Map
```bash
python -m loader.saliency_generator --data rotc --arch meddef1 --depth 1.0 --model_path "out/normal_training/rotc/meddef1_1.0/adv/save_model/best_meddef1_1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth"  --image_path "out/normal_training/chest_xray/resnet_18/attack/fgsm/sample_0_adv.png"



python -m loader.saliency_generator --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --image_paths "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_orig.png" "out/normal_training/rotc/densenet_121/attack/fgsm/sample_3_orig.png" "out/normal_training/rotc/densenet_121/attack/fgsm/sample_4_orig.png"
```
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


```bash
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --optimizer sgd --scheduler StepLR

python main.py --data chest_xray --arch resnet --depth '{"resnet": [18]}' --train_batch 32 --epochs 100 --lr 0.001 --optimizer sgd --scheduler StepLR --patience 30

python main.py --experiment_mode --enable optim --data chest_xray --arch resnet --depth '{"resnet": [18]}' --train_batch 32 --epochs 100 --lr 0.01 --optimizer sgd --scheduler CosineAnnealingLR --min_lr 1e-6 --patience 30

python main.py --experiment_mode --enable optim reg --data chest_xray --arch resnet --depth '{"resnet": [18]}' --train_batch 32 --epochs 100 --lr 0.01 --optimizer sgd --scheduler CosineAnnealingWarmRestarts --cycle_length 20 --drop 0.3 --weight_decay 5e-5 --patience 30 --min_epochs 50

python main.py --experiment_mode --enable optim reg --data chest_xray --arch resnet --depth '{"resnet": [18]}' --train_batch 32 --epochs 100 --lr 0.01 --optimizer sgd --scheduler StepLR --drop 0.3 --weight_decay 5e-5 --early_stopping_metric accuracy --patience 30

```



```bash
# Standard cross-entropy (default)
python main.py --data chest_xray --arch resnet --loss_type standard --drop 0.3 --weight_decay 1e-3

# Automatic class weighting based on distribution
python main.py --data chest_xray --arch resnet --loss_type weighted --drop 0.3 --weight_decay 1e-3

# Aggressive weighting for severe imbalance
python main.py --data chest_xray --arch resnet --loss_type aggressive  --drop 0.3 --weight_decay 1e-3

# Dynamic weighting that focuses on harder examples over time
python main.py --data chest_xray --arch resnet --loss_type dynamic --focal_alpha 0.5 --focal_gamma 2.0

```

```bash
# Tensorboard

tensorboard --logdir=out/runs


```

```bash
python main.py --data chest_xray --task_name normal_training --epochs 100 --train_batch 16 --lr 0.0001 --drop 0.2 --gpu-ids 0 --arch transformer --depth '{"transformer": ["tiny"]}' --pin_memory --weight_decay 1e-4

## Handle class imbalance
python main.py --data chest_xray --task_name normal_training --loss_type weighted --use_weighted_sampler --epochs 100 --arch meddef1 --depth '{"meddef1": ["1.0"]}' --early_stopping_metric f1

python main.py --data chest_xray --task_name normal_training --loss_type weighted --epochs 100 --train_batch 16 --lr 0.0001 --drop 0.2 --gpu-ids 0 --arch meddef1 --depth '{"meddef1": ["1.0"]}' --pin_memory --weight_decay 1e-4 --early_stopping_metric accuracy


python main.py --data ccts --task_name normal_training --epochs 100 --train_batch 16 --lr 0.0001 --drop 0.2 --gpu-ids 0 --arch meddef1 --depth '{"meddef1": ["1.0"]}' --pin_memory --weight_decay 1e-4

### Data preprocessing
python dataset_processing.py --datasets chest_xray --enforce_split --train_split 0.8 --val_split 0.1 --test_split 0.1


python dataset_processing.py --datasets ccts --enforce_split --train_split 0.8 --val_split 0.1 --test_split 0.1


```


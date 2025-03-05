<<<<<<< HEAD
# meddef1
=======
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

## When running with config.yaml configuration overriding arg_parser

```bash
python main.py --data ccts --data_key ccts --task_name normal_training --epochs 2 --train_batch 8 --test-batch 8 --lr 0.001 --drop 0.5 --gpu-ids 0 --arch [resnet, meddef1_] --depth "{'resnet': [18,34], 'meddef1_': [1.0,1.1]}"
```

```bash
python main.py --data ccts --data_key ccts --task_name normal_training --epochs 10 --train_batch 8 --test-batch 8 --lr 0.001 --min-lr 1e-6 --warmup-epochs 5 --lr-scheduler cosine --weight-decay 1e-5 --lambda_l2 0.0001 --drop 0.2 --gpu-ids 0 --arch meddef2_ --depth "{'meddef2_': [2.1]}" --adv_training
```

```bash
 python main.py --data ccts --data_key ccts --task_name normal_training --epochs 100 --train_batch 32 --test-batch 32 --lr 0.01 --min-lr 1e-7 --warmup-epochs 10 --lr-scheduler cosine --weight-decay 1e-6 --lambda_l2 0.0001 --drop 0.2 --gpu-ids 0 --arch meddef2_ --depth "{'meddef2_': [2.0, 2.1, 2.2, 2.3, 2.4]}" --adv_training --max-grad-norm 1.0 --accumulation_steps 4 --patience 15
```

```bash
python main.py --data ccts --data_key ccts --arch resnet --depth "{'resnet': [18]}" --train_batch 32 --epochs 2 --lr 0.001 --drop 0.3 --accumulation_steps 8 --workers 1 --pin_memory
```

```bash
python main.py --data ccts --data_key ccts --arch meddef1_ --depth "{'meddef1_': [1.0, 1.1, 1.2]}" --train_batch 64 --epochs 2 --lr 0.001 --drop 0.3 --workers 8 --pin_memory
```




```bash
python dataset_processing.py --data ccts --output_dir processed_data

python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0, 1.1, 1.2]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam

python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0, 1.1, 1.2]}' --train_batch 32 --epochs 2 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --scheduler StepLR


python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.2]}' --train_batch 64 --epochs 100 --lr 0.0001 --drop 0.3 --weight_decay 0.0001 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --attack_eps 0.2 --adv_weight 0.3 --attack_type pgd

## 

python main.py --data rotc --arch resnext --depth '{"resnext": [50]}' --train_batch 64 --epochs 2 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam


### test

python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch32_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"


python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch32_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"

python test.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0]}' --test_batch 32 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --model_path "out/normal_training/ccts/meddef1__1.0/save_model/best_meddef1__1.0_ccts_epochs5_lr0.001_batch32_20250217.pth"


```


Adversarial Traiing
```bash
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1_": [1.0, 1.1, 1.2]}' --train_batch 32 --epochs 3 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm


python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0, 1.1, 1.2]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial 



```

```bash
# Step 1: Generate and save attacks
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

# Step 2: Train with pre-generated attacks
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 5 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm


```


### defense
```bash
### test before prune -> normal image normal train
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/adv/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250224.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"

### test before prune -> adversarial image normal train
## 1 | Confidence  1.0000
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/b_meddef1__1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" --image_path "out/normal_training/rotc/meddef1__1.0/attack/bim+jsma/sample_0_orig.png"
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
python test.py --data chest_xray --arch meddef1_ --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1__1.0/adv/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch32_20250303.pth" --image_path "out/normal_training/chest_xray/meddef1__1.0/attack/fgsm/sample_0_adv.png"

```
python main.py --data chest_xray --arch meddef1 --depth '{"meddef1": [1.0, 1.1,1.2]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm bim jsma pgd


python main.py --data chest_xray --arch vgg --depth '{"vgg": [16]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm

python main.py --data chest_xray --arch densenet --depth '{"densenet": [121]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm

python main.py --data chest_xray --arch resnet --depth '{"resnet": [18,34]}' --train_batch 32 --epochs 100 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 1 --task_name normal_training --optimizer adam --adversarial --attack_eps 0.2 --adv_weight 0.5 --attack_type fgsm

```bash

Resnet18	11.18	86.57	88.33	86.57	86.29
Densenet121	85.84	90.59	85.84	84.73     84.73
MedDef1.0	21.84	99.27	99.29	99.27	99.27	
VGG16       133.49  89.54  89.99   89.54  89.54
Resnet34    21.29  91.59  90.99   91.59  91.59




### test before prune -> normal image | adversarial train


## pruning
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name defense --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" --prune_rate 0.3


### test before prune
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"


### test adversarial
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/defense/rotc/meddef1__1.0/save_model/pruned_meddef1__1.0_epochs100_lr0.001_batch32_20250224.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg" --task_name defense


```

### Robustness Test
```bash
#### Evaluate a single model against multiple attacks and pruning rates:
python evaluate_attacks.py --data chest_xray --arch meddef1_ --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1__1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" --attack_types fgsm pgd bim --attack_eps 0.1 --prune_rates 0.0 0.1 0.3 0.5 0.7 --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 1

#### Compare a specific attack at a single pruning rate:
python evaluate_attacks.py --data chest_xray --arch meddef1_ --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1__1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" --attack_types fgsm --attack_eps 0.1 --prune_rates 0.3 --gpu-ids 1

#### testing different models

# For ResNet model
python evaluate_attacks.py --data chest_xray --arch resnet  --depth 18 --model_path "out/normal_training/chest_xray/resnet_18/save_model/best_resnet_18_chest_xray_epochs100_lr0.001_batch32_20250227.pth" --attack_types fgsm pgd --prune_rates 0.0 0.3 0.5 --gpu-ids 1

# For MedDef model
python evaluate_attacks.py --data chest_xray --arch meddef1_ --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1__1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth" --attack_types fgsm pgd --prune_rates 0.0 0.3 0.5 --gpu-ids 1

# For Densenet
python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types fgsm pgd bim --attack_eps 0.2 --prune_rates 0.0 0.1 0.3 0.5 0.7 --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 1

python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types cw zoo boundary elasticnet onepixel fgsm pgd bim jsma --attack_eps 0.2 --prune_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 32 --num_workers 4 --pin_memory --gpu-ids 1


python evaluate_attacks.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --attack_types fgsm --attack_eps 0.2 --prune_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1


python evaluate_attacks.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/b_meddef1__1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" --attack_types fgsm --attack_eps 0.2 --prune_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1

python evaluate_attacks.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/b_meddef1__1.0/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch32_20250301.pth" --attack_types fgsm pgd bim jsma --attack_eps 0.2 --prune_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 --batch_size 64 --num_workers 4 --pin_memory --gpu-ids 1

```


### Salient Map
```bash
python generate_saliency_maps.py --data chest_xray --arch meddef1_ --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1__1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0001_batch16_20250227.pth"  --image_path "out/normal_training/chest_xray/resnet_18/attack/fgsm/sample_0_adv.png"


python generate_saliency_maps.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth"  --image_path "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_adv.png"



python generate_saliency_maps.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --image_path "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_adv.png"



python generate_saliency_maps.py --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --image_paths "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_orig.png" "out/normal_training/rotc/densenet_121/attack/fgsm/sample_2_orig.png" "out/normal_training/rotc/densenet_121/attack/fgsm/sample_2_orig.png"


python -m loader.saliency_generator --data rotc --arch densenet --depth 121 --model_path "out/normal_training/rotc/densenet_121/adv/save_model/best_densenet_121_rotc_epochs100_lr0.0001_batch32_20250228.pth" --image_paths "out/normal_training/rotc/densenet_121/attack/fgsm/sample_0_orig.png" "out/normal_training/rotc/densenet_121/attack/fgsm/sample_3_orig.png" "out/normal_training/rotc/densenet_121/attack/fgsm/sample_4_orig.png"
```
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


>>>>>>> 9c762dfc530c4b960474568a87004c0d4481da5f

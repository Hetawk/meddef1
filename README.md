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
 python main.py --data ccts --data_key ccts --task_name normal_training --epochs 100 --train_batch 32 --test-batch 32 --lr 0.01 --min-lr 1e-7 --warmup-epochs 10 --lr-scheduler cosine --weight-decay 1e-6 --lambda_l2 0.00001 --drop 0.2 --gpu-ids 0 --arch meddef2_ --depth "{'meddef2_': [2.0, 2.1, 2.2, 2.3, 2.4]}" --adv_training --max-grad-norm 1.0 --accumulation_steps 4 --patience 15
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


test

python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch32_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"


python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch32_20250221.pth" --image_path "processed_data/rotc/test/NORMAL/NORMAL-9251-1.jpeg"

python test.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0]}' --test_batch 32 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --model_path "out/normal_training/ccts/meddef1__1.0/save_model/best_meddef1__1.0_ccts_epochs5_lr0.001_batch32_20250217.pth"


```


Adversarial Traiing
```bash
python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 3 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm


python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0, 1.1, 1.2]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial 



```

```bash
# Step 1: Generate and save attacks
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

# Step 2: Train with pre-generated attacks
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 5 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm
```



## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.



#!/bin/bash

# Activate virtual environment if needed
# source ./venv/bin/activate

### MedDef
python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 64 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

python main.py --data chest_xray --arch meddef1_ --depth '{"meddef1_": [1.0}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

### Resnet

python main.py --data rotc --arch resnet --depth '{"resnet": [18]}' --train_batch 64 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

python main.py --data ccts --arch resnet --depth '{"resnet": [18]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

python main.py --data chest_xray --arch resnet --depth '{"resnet": [18}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm


### Densenet
python main.py --data rotc --arch densenet --depth '{"densenet": [121]}' --train_batch 64 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

python main.py --data ccts --arch densenet --depth '{"densenet": [121]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

python main.py --data chest_xray --arch densenet --depth '{"densenet": [121}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm




# python main.py --data tbcr --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

# python main.py --data scisic --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam --adversarial --attack_type fgsm

#
## train on rotc
#python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
#
## train on dermnet
#python main.py --data dermnet --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
## train on scisic
#python main.py --data scisic --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
#
## train on chest_xray
#python main.py --data chest_xray --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
#
## train on tbrc
#python main.py --data tbcr --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
#
### Example command to process dataset 'dermnet'
##python dataset_processing.py --data rotc --output_dir processed_data
#
## # # run training with a specified model and dataset
###  python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --train_batch 32 --epochs 100 --lr 0.001 --drop 0.3 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
## python test.py --data ccts --arch meddef1_ --depth 1.0 --test_batch 32 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training
##
##
### run test
##python test.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0]}' --test_batch 32 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training
### # Add further commands as required

echo "All commands executed."

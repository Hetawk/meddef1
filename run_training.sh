#!/bin/bash

# Set default values
DATA="chest_xray"
ARCH="meddef1_"
DEPTH='{"meddef1_": [1.0]}'
GPU_ID=1
TASK_NAME="normal_training"
OPTIMIZER="adam"
NUM_WORKERS=4

# Function to run training with specific parameters
run_training() {
    local lr=$1
    local batch_size=$2
    local dropout=$3
    local weight_decay=$4
    local epochs=$5
    
    echo "Running with: lr=${lr}, batch=${batch_size}, dropout=${dropout}, weight_decay=${weight_decay}, epochs=${epochs}"
    
    python main.py \
        --data ${DATA} \
        --arch ${ARCH} \
        --depth "${DEPTH}" \
        --train_batch ${batch_size} \
        --epochs ${epochs} \
        --lr ${lr} \
        --drop ${dropout} \
        --weight_decay ${weight_decay} \
        --num_workers ${NUM_WORKERS} \
        --pin_memory \
        --gpu-ids ${GPU_ID} \
        --task_name "${TASK_NAME}"\
        --optimizer ${OPTIMIZER}
}

# Choose which experiment to run
case "$1" in
    "baseline")
        # Your current parameters
        run_training 0.0001 16 0.3 0.0 100
        ;;
    "higher_dropout")
        # Try increasing dropout to combat overfitting
        run_training 0.0001 16 0.5 0.0 100
        ;;
    "with_regularization")
        # Add weight decay (L2 regularization)
        run_training 0.0001 16 0.3 0.0001 100
        ;;
    "larger_batch")
        # Try larger batch size
        run_training 0.0001 32 0.3 0.0 100
        ;;
    "smaller_batch")
        # Try smaller batch size
        run_training 0.0001 8 0.3 0.0 100
        ;;
    "higher_lr")
        # Try higher learning rate
        run_training 0.0003 16 0.3 0.0 100
        ;;
    "full_regularization")
        # Combine higher dropout with weight decay
        run_training 0.0001 16 0.5 0.0001 100
        ;;
    "all")
        # Run all experiments sequentially
        run_training 0.0001 16 0.3 0.0 100
        run_training 0.0001 16 0.5 0.0 100
        run_training 0.0001 16 0.3 0.0001 100
        run_training 0.0001 32 0.3 0.0 100
        run_training 0.0001 8 0.3 0.0 100
        run_training 0.0003 16 0.3 0.0 100
        run_training 0.0001 16 0.5 0.0001 100
        ;;
    *)
        echo "Usage: $0 [baseline|higher_dropout|with_regularization|larger_batch|smaller_batch|higher_lr|full_regularization|all]"
        echo "Example: $0 higher_dropout"
        exit 1
        ;;
esac

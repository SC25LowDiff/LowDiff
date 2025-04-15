#!/bin/zsh

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=cifar100
MODEL=resnet50
EPOCHS=10
BATCH_SIZE=64
FREQ=10
SAVE_DIR=/save_dir
RESUME=0

# Distributed training with DeepSpeed
deepspeed --hostfil=hostfile ./torch/checkfreq.py \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --freq $FREQ \
  --save-dir $SAVE_DIR \
  --resume $RESUME

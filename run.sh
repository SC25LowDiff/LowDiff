#!/bin/zsh

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=imagenet
MODEL=resnet101
EPOCHS=10
BATCH_SIZE=64
COMPRESSOR=topk
COMPRESS_RATIO=0.01
DIFF=false
FREQ=0
PACK=1

# Distributed training with DeepSpeed
deepspeed --hostfile=hostfile cv.py \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --compressor $COMPRESSOR \
  --compressor_ratio $COMPRESS_RATIO \
  --diff $DIFF \
  --freq $FREQ \
  --pack $PACK

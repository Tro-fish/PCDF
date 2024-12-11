#!/bin/bash

# Hyperparameters and paths
RUN_NAME="PaliGemma_Example"
TRAIN_PATH="dataset/chart2text_statista/chart2text_statista_train.json"
VALID_PATH="dataset/chart2text_statista/chart2text_statista_val.json"
BATCH_SIZE=4
LEARNING_RATE=3e-5
NUM_STEPS=10000
SAVE_STEPS=500

# Execute the Python script
python -m torch.distributed.launch --nproc_per_node=2 train_paligemma.py \
  --run_name $RUN_NAME \
  --train_path $TRAIN_PATH \
  --valid_path $VALID_PATH \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --num_steps $NUM_STEPS \
  --save_steps $SAVE_STEPS
#!/usr/bin bash

python run.py \
    --backbone bert \
    --method ADB \
    --seed 0 \
    --lr 2e-5 \
    --dataname semeval \
    --task_type relation_detection \
    --max_length 120 \
    --optim adamw \
    --gpu_id 0 \
    --labeled_ratio 1.0 \
    --known_cls_ratio 0.25 \
    --num_train_epochs 10 
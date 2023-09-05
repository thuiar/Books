#!/usr/bin bash

datanames=('semeval' 'wiki80')
# datanames=('wiki20m')
for dataname in ${datanames[@]}
do
    for kcr in 0.25 0.5 0.75
    do
        python run.py \
            --backbone bert \
            --method CDACPlus \
            --method_type semi_supervised \
            --seed 0 \
            --lr 2e-5 \
            --dataname $dataname \
            --task_type relation_discovery \
            --max_length 120 \
            --optim adamw \
            --gpu_id 0 \
            --labeled_ratio 1.0 \
            --known_cls_ratio $kcr
    done
done
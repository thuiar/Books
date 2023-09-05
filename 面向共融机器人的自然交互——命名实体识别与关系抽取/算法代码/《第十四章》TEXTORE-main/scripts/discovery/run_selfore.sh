#!/usr/bin bash

datanames=('wiki80')
for s in 0
do
    for dataname in ${datanames[@]}
    do
        python run.py \
            --backbone bert \
            --method SelfORE \
            --method_type unsupervised \
            --seed $s \
            --lr 2e-5 \
            --dataname $dataname \
            --task_type relation_discovery \
            --max_length 120 \
            --optim adamw \
            --gpu_id 1 \
            --labeled_ratio 1.0 \
            --known_cls_ratio 0.25 \
            --train_model 1 \
            --freeze_bert_parameters 1
    done
done
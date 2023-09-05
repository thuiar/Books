#!/usr/bin bash

datanames=('semeval' 'wiki80')
for s in 0
do
    for dataname in ${datanames[@]}
    do
        python run.py \
            --backbone bert \
            --method ODC \
            --method_type unsupervised \
            --seed $s \
            --lr 0.1 \
            --dataname $dataname \
            --task_type relation_discovery \
            --max_length 120 \
            --optim sgd \
            --gpu_id 1 \
            --labeled_ratio 1.0 \
            --known_cls_ratio 0.25 \
            --freeze_bert_parameters 1
    done
done
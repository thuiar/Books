#!/usr/bin bash

datanames=('semeval')
for s in 0 1 2 3 4
do
    for dataname in ${datanames[@]}
    do
        for kcr in 0.25 0.5 0.75
        do
            python run.py \
                --backbone bert \
                --method MSP \
                --seed $s \
                --lr 2e-5 \
                --dataname $dataname \
                --task_type relation_detection \
                --max_length 120 \
                --optim adamw \
                --gpu_id 0 \
                --labeled_ratio 1.0 \
                --known_cls_ratio $kcr
        done
    done
done
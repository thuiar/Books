#!/usr/bin bash

datanames=('semeval' 'wiki80')
for s in 0 1 2 3 4
do
    for dataname in ${datanames[@]}
    do
        for kcr in 0.25 0.5 0.75
        do
            python run.py \
                --backbone bert \
                --method RSN \
                --method_type semi_supervised \
                --seed $s \
                --lr 2e-5 \
                --dataname $dataname \
                --task_type relation_discovery \
                --max_length 120 \
                --optim adam \
                --gpu_id 1 \
                --labeled_ratio 1.0 \
                --batch_size 2 \
                --batch_num 5000 \
                --known_cls_ratio $kcr \
                --select_cluster kmean \
                --train_model 1
        done
    done
done
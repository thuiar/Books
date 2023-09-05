#!/usr/bin bash

datanames=('wiki80')
# datanames=('wiki20m')
for s in 0
do
    for dataname in ${datanames[@]}
    do
        for kcr in 0.25
        do
            python run.py \
                --backbone bert \
                --method DTGNS \
                --method_type semi_supervised \
                --seed $s \
                --lr 1e-4 \
                --dataname $dataname \
                --task_type relation_discovery \
                --max_length 120 \
                --optim adamw \
                --gpu_id 0 \
                --labeled_ratio 1.0 \
                --known_cls_ratio $kcr \
                --train_model 1 \
                --warmup_proportion 0.1 \
                --use_pretrain 0 \
                --this_name un_pretrain \
                --loop_nums 100
                
        done
    done
done
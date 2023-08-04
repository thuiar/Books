#!/usr/bin bash

for dataset in 'stackoverflow' 'oos' 'banking'
do
    for known_cls_ratio in 0.25 0.5 0.75 
    do
        for labeled_ratio in 1.0
        do
            for seed in 0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'ADB' \
                --log_id $seed \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert' \
                --config_file_name 'ADB' \
                --gpu_id '0' \
                --train \
                --save_results \
                --results_file_name 'results_ADB.csv' \
                --save_frontend_results
            done
        done
    done
done
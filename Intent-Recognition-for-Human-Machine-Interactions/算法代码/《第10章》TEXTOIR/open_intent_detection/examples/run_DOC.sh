#!/usr/bin bash

for dataset in 'banking' 'oos' 'stackoverflow'
do
    for known_cls_ratio in 0.25 0.5 0.75
    do
        for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
        do
            for seed in 0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'DOC' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_doc' \
                --config_file_name 'DOC' \
                --loss_fct 'Binary_CrossEntropyLoss' \
                --train \
                --gpu_id '0' \
                --save_results \
                --results_file_name 'results_DOC.csv' \
                --save_model
            done
        done
    done
done

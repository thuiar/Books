import subprocess
import os
seeds = ['42','442','4422','100','3']
known_cls_ratios = ['0.5']
labeled_ratios = ['1.0']
methods = [ 'ours.py']
datasets = ['oos']
alphas = [ '0.35', '0.45']
lambs = ['0.05']

max_seq_lengths = {'oos':'30', 'fewrel':'40', 'stackoverflow':'45'}
for seed in seeds:
    print(seed, "start")
    for dataset in datasets:
        for known_cls_ratio in known_cls_ratios:
            for labeled_ratio in labeled_ratios:
                for method in methods:
                    for alpha in alphas:
                        for lamb in lambs:
                            command = [
                                'python', method,
                                '--data_dir', os.path.join('data',dataset),
                                '--task_name', dataset,
                                '--max_seq_length', max_seq_lengths[dataset],
                                '--known_cls_ratio',known_cls_ratio,
                                '--labeled_ratio',labeled_ratio,
                                '--method',method,
                                '--seed',seed,
                                '--gpu_id','2',
                                '--lamb',lamb,
                                '--alpha',alpha,
                                #######################################################
                                '--num_train_epochs','100',
                                '--learning_rate','2e-5',
                                '--train_batch_size', '64',
                                '--eval_batch_size','64',
                                '--feat_dim','768',
                                '--threshold','0.5',
                                '--save_results_path','data/results', 
                            ]
                            p = subprocess.Popen(command)
                            p.communicate()
                            print(seed, "finish, return code=", p.returncode)
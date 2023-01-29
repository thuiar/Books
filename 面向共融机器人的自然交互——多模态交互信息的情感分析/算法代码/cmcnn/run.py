import os
import gc
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob

from trains import *
from config import *
from utils.log import *
from utils.metricsTop import *
from utils.functions import *
from utils.gpu_memory_log import gpu_memory_log
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import FERDataLoader 
from sklearn.model_selection import KFold


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists(args.res_save_path):
        os.makedirs(args.res_save_path)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    status("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % args.gpu_ids[0] if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = FERDataLoader(args)
    model = AMIO(args).to(device)
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load best model
    model_save_pathes = glob(os.path.join(args.model_save_path,\
                                            f'{args.modelName}-{args.datasetName}.pth'))
    assert len(model_save_pathes) == 1
    model.load_state_dict(torch.load(model_save_pathes[0]))
    # torch.save(model.cpu(), './results/save_models/MTL_MRN.pkl')
    # do test
    mode = "test"
    results = atio.do_test(model, dataloader, mode=mode)
    # gpu_memory_log()
    return results


def run_debug(seeds, debug_times=200):
    has_debuged = [] # save used paras
    args = parse_args()
    config = ConfigDebug(args)
    save_file_path = os.path.join(args.res_save_path, args.datasetName+'-debug', args.modelName+'.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    for i in range(debug_times):
        # cancel random seed
        setup_seed(int(time.time()))
        args = config.get_config()
        args.knum = -1
        # print debugging params
        print("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, debug_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                print(k, ':', v)
        print("#"*90)
        status('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            print('These paras have been used!')
            time.sleep(3)
            continue
        try:
            has_debuged.append(cur_paras)
            results = []
            for j, seed in enumerate(seeds):
                args.cur_time = j + 1
                setup_seed(seed)
                results.append(run(args))
        except Exception as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            print(e)
            continue
        # save results to csv
        status('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        tmp = []
        for col in list(df.columns):
            if col in args.d_paras:
                tmp.append(args[col])
            elif col in results[0].keys():
                values = [r[col] for r in results]
                tmp.append(round(sum(values) / len(values), 4))
        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        status('Results are saved to %s...' %(save_file_path))


def run_normal(seeds):
        # cm_save = []
    if True:
        args = parse_args()
        # args.modelName = model_name
        # load config
        config = Config(args)
        args = config.get_config()
        model_results = []
        args.knum = -1
        # run results
        for i, seed in enumerate(seeds):
            args.cur_time = i+1
            setup_seed(seed)
            args.seed = seed
            print(args)
            status('Start running %s...' %(args.modelName))
            # runnning
            test_results = run(args)
            # restore results
            # cm_save.append(test_results.pop('CM').tolist())
            model_results.append(test_results)
        # save results
        criterions = list(model_results[0].keys())
        columns = ["Model"]
        columns += ['Avg-'+c for c in criterions]
        for seed in seeds:
            columns += [str(seed)+'-'+c for c in criterions]
        df = pd.DataFrame(columns=columns)
        res = [args.modelName]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 4)
            std = round(np.std(values)*100, 4)
            res.append((mean, std))
        for m in model_results:
            res += [round(m[c] * 100, 4) for c in criterions]
        df.loc[len(df)] = res
        save_path = os.path.join(args.res_save_path, args.datasetName, \
                                    args.modelName + '.csv')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        df.to_csv(save_path, index=None)
        status('Results are saved to %s...' %(save_path))
        # save 
        # save_path = os.path.join(args.res_save_path, args.datasetName, \
        #                             args.modelName + '.npy')
        # np.save(save_path, np.array(cm_save))
   

def run_val(seeds):
    cm_save = []
    # if True:
    args = parse_args()
    # args.datasetName = cur_model
    save_path = os.path.join(args.res_save_path, args.datasetName, \
                                    args.modelName + '.csv')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # if os.path.exists(save_path):
    #     os.remove(save_path) # start a new one
    # load config
    config = Config(args)
    args = config.get_config()
    train_df0 = pd.read_csv(os.path.join(args.label_dir,'train.csv'))
    kf = KFold(10,shuffle = False)
    Avg_row = []
    try:
        for knum, indexs in enumerate(kf.split(train_df0)):
            args.knum = knum + 1
            # print(indexs)
            args.train_df = train_df0.iloc[indexs[0],:]
            args.test_df = train_df0.iloc[indexs[1],:]
            model_results = []
            cm_tmp = []
            # run results
            for i, seed in enumerate(seeds):
                args.cur_time = i+1
                args.seed = seed
                setup_seed(seed)
                # runnning
                status('Start running %s...' %(args.modelName))
                # print(args)
                test_results = run(args)
                break
                cm_tmp.append(test_results.pop('CM').tolist())
                # restore results
                model_results.append(test_results)
            break
            cm_save.append(cm_tmp)
            # save results
            criterions = list(model_results[0].keys())
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
            else:
                columns = ["Model"] + ['Avg-'+c for c in criterions]
                for seed in seeds:
                    columns += [str(seed)+'-'+c for c in criterions]
                df = pd.DataFrame(columns = columns)
            res = [args.modelName]
            cur_avg_row = []
            for c in criterions:
                values = [r[c] for r in model_results]
                mean = round(np.mean(values)*100, 4)
                std = round(np.std(values)*100, 4)
                res.append((mean, std))
                cur_avg_row.append(mean)
            for m in model_results:
                cur_res = [round(m[c] * 100, 4) for c in criterions]
                res += cur_res
                cur_avg_row += cur_res
            Avg_row.append(cur_avg_row)
            df.loc[len(df)] = res
            df.to_csv(save_path, index=None)
            status('Results are saved to %s...' %(save_path))
    except Exception as e:
        print(e)
    finally:
        # compute average values
        df = pd.read_csv(save_path)
        Avg_row = np.array(Avg_row).mean(axis=0)
        df.loc[len(df)] = ['AVG'] + list(Avg_row)
        df.to_csv(save_path, index=None)
        # save 
        save_path = os.path.join(args.res_save_path, args.datasetName, \
                                    args.modelName + '.npy')
        np.save(save_path, np.array(cm_save))


def run_debug_val(seeds, debug_times=200):
    has_debuged = [] # save used paras
    args = parse_args()
    save_path = os.path.join(args.res_save_path, args.datasetName+'-debug', \
                                    args.modelName + '.csv')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # load config
    config = ConfigDebug(args)
    for i in range(debug_times):
        setup_seed(int(time.time()))
        args = config.get_config()
        # print debugging params
        print("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, debug_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                print(k, ':', v)
        print("#"*90)
        # restore existed paras
        if i != 0 and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            print('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        # split dataset
        train_df0 = pd.read_csv(os.path.join(args.label_dir,'train.csv'))
        kf = KFold(10, shuffle=False)
        Avg_row = []
        try:
            for knum, indexs in enumerate(kf.split(train_df0)):
                # if(knum >= 5):
                #     break
                args.knum = knum + 1
                args.train_df = train_df0.iloc[indexs[0],:]
                args.test_df = train_df0.iloc[indexs[1],:]
                model_results = []
                # print(indexs)
                # run results
                for j, seed in enumerate(seeds):
                    args.cur_time = j+1
                    args.seed = seed
                    setup_seed(seed)
                    # runnning
                    status('Start running %s...' %(args.modelName))
                    # print(args)
                    test_results = run(args)
                    # restore results
                    model_results.append(test_results)
                # save results
                criterions = list(model_results[0].keys())
                if not os.path.exists(save_path):
                    columns = [k for k in args.d_paras] + [c for c in criterions]
                    df = pd.DataFrame(columns = columns)
                    df.to_csv(save_path, index=None)
                cur_avg_row = []
                for c in criterions:
                    values = [r[c] for r in model_results]
                    mean = round(np.mean(values)*100, 4)
                    cur_avg_row.append(mean)
                Avg_row.append(cur_avg_row)
            # compute average values
            df = pd.read_csv(save_path)
            Avg_row = np.array(Avg_row).mean(axis=0)
            df.loc[len(df)] = [args[col] for col in list(df.columns) if col in args.d_paras] + list(Avg_row)
            df.to_csv(save_path, index=None)
        except Exception as e:
            print(e)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', type=bool, default=False, 
                        help='debug parameters ?')
    parser.add_argument('--tf_mode', type=bool, default=False,
                        help='is transfer test ?')
    parser.add_argument('--val_mode', type=bool, default=False,
                        help='10 folds cross validation ?')
    parser.add_argument('--modelName', type=str, default='FER_DCNN',
                        help='support FER_DCNN/LM_DCNN/CMCNN')
    parser.add_argument('--datasetName', type=str, default='RAF',
                        help='support RAF/SFEW2/CK+/OULU_CASIA/MMI')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='results',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()


if __name__ == '__main__':
    seeds = [1, 12, 123, 1234, 12345]
    tmp_args = parse_args()
    if tmp_args.debug_mode and tmp_args.val_mode:
        run_debug_val(seeds, debug_times=200)
    elif tmp_args.debug_mode:
        run_debug(seeds, debug_times=200)
    elif tmp_args.val_mode:
        run_val(seeds)
    else:
        run_normal(seeds)
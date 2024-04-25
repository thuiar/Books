import os
import gc
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from trains import *
from config import *
from utils.log import *
from utils.metricsTop import *
from utils.functions import *
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import FERDataLoader 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def do_test_stl(args, model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    metrics = MetricsTop(args).getMetrics(args.metricsName)
    with torch.no_grad():
        for batch_data in tqdm(dataloader['train']):
            data = batch_data['data'].to(args.device)
            labels = batch_data['labels'].to(args.device)
            emotions = batch_data['emotions']
            # model
            output  = model(data)
            y_pred.append(output['fer_output'].cpu())
            y_true.append(labels.cpu())

        for batch_data in tqdm(dataloader['test']):
            data = batch_data['data'].to(args.device)
            labels = batch_data['labels'].to(args.device)
            emotions = batch_data['emotions']
            # model
            output  = model(data)
            y_pred.append(output['fer_output'].cpu())
            y_true.append(labels.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    print("TF-(%s)" % args.modelName, dict_to_str(results))
    return results

def do_test_mtl(args, model, dataloader):
    model.eval()
    fer_y_pred, fer_y_true = [], []
    lm_y_pred, lm_y_true = [], []
    metrics = MetricsTop(args).getMetrics(args.metricsName)
    with torch.no_grad():
        for batch_data in tqdm(dataloader['train']):
            data = batch_data['data'].to(args.device)
            fer_labels = batch_data['labels'].to(args.device)
            lm_labels = batch_data['landmarks'].to(args.device)
            emotions = batch_data['emotions']
            # model
            output = model(data)
            fer_y_pred.append(output['fer_output'].cpu())
            fer_y_true.append(fer_labels.cpu())
            lm_y_pred.append(output['lm_output'].cpu())
            lm_y_true.append(lm_labels.cpu())

        for batch_data in tqdm(dataloader['test']):
            data = batch_data['data'].to(args.device)
            fer_labels = batch_data['labels'].to(args.device)
            lm_labels = batch_data['landmarks'].to(args.device)
            emotions = batch_data['emotions']
            # model
            output = model(data)
            fer_y_pred.append(output['fer_output'].cpu())
            fer_y_true.append(fer_labels.cpu())
            lm_y_pred.append(output['lm_output'].cpu())
            lm_y_true.append(lm_labels.cpu())

    fer_pred, fer_true = torch.cat(fer_y_pred), torch.cat(fer_y_true)
    lm_pred, lm_true = torch.cat(lm_y_pred), torch.cat(lm_y_true)
    results = metrics(fer_pred, fer_true, lm_pred,lm_true)
    print("TF-(%s)" % args.modelName, dict_to_str(results))
    return results

def run(args):
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    if not os.path.exists(args.res_save_path):
        os.mkdir(args.res_save_path)
    args.knum = -1
    for datasets in [['RAF', 'CK+'],['CK+', 'RAF'],['CK+','OULU_CASIA'], ['OULU_CASIA','CK+'], ['RAF','SFEW2'], ['SFEW2','RAF']]:
        args.srcDatasetName = datasets[0]
        args.dstDatasetName = datasets[1]
        results = []
        for i, seed in enumerate(args.seeds):
            args.cur_time = i+1
            setup_seed(seed)
            # get dst dataset params
            args.datasetName = args.dstDatasetName
            config = Config(args)
            dst_args = config.get_config()
            # get src dataset params
            args.datasetName = args.srcDatasetName
            config = Config(args)
            src_args = config.get_config()
            # get dataloader
            src_dataloader = FERDataLoader(src_args)
            dst_dataloader = FERDataLoader(dst_args)
            # gpu
            using_cuda = len(src_args.gpu_ids) > 0 and torch.cuda.is_available()
            device = torch.device('cuda:%d' % src_args.gpu_ids[0] if using_cuda else 'cpu')
            src_args.device = device
            # build model
            model = AMIO(src_args).to(device)
            atio = ATIO().getTrain(src_args)
            # do train
            atio.do_train(model, src_dataloader)
            # load best model
            model_save_pathes = glob(os.path.join(args.model_save_path,\
                                                    f'{args.modelName}-{args.datasetName}.pth'))
            assert len(model_save_pathes) == 1
            model.load_state_dict(torch.load(model_save_pathes[0]))
            # do test
            if src_args.modelName == 'FER_DCNN':
                results.append(do_test_stl(src_args, model, dst_dataloader))
            else:
                results.append(do_test_mtl(src_args, model, dst_dataloader))
        # save results
        df_path = os.path.join(args.res_save_path, 'TFResults-6.csv')
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            columns = ['ModelName', 'SrcDataset', 'DstDataset', 'Avg_Total_Acc', 'Avg_Average_Acc', 'Avg_Macro_F1']
            for s in args.seeds:
                columns += [str(s)+'-'+r for r in ['Total_Acc', 'Average_Acc', 'Macro_F1']]
            df = pd.DataFrame(columns=columns)
        res = []
        for r in results:
            res.append([r[k] for k in ['Total_Acc', 'Average_Acc', 'Macro_F1']])
        res = np.array(res)
        final_res = res.mean(axis=0).tolist() + res.reshape(-1).tolist()
        df.loc[len(df)] = [args.modelName, args.srcDatasetName, args.dstDatasetName] + final_res
        df.to_csv(df_path, index=None)
        print('Results are saved to %s...' %(df_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_mode', type=bool, default=True,
                        help='is transfer test ?')
    parser.add_argument('--modelName', type=str, default='Our',
                        help='support FER_DCNN/CMCNN')
    parser.add_argument('--srcDatasetName', type=str, default='RAF',
                        help='support RAF/SFEW2/CK+/OULU_CASIA')
    parser.add_argument('--dstDatasetName', type=str, default='CK+',
                        help='support RAF/SFEW2/CK+/OULU_CASIA')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='results/TFModels2',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results/TFMetrics2',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.seeds = [1, 12, 123, 1234, 12345]
    run(args)
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
from sklearn.model_selection import KFold

def do_test(args, model, dataloader, mode='test'):
    model.eval()
    y_true, y_pred = [], []
    metrics = MetricsTop(args).getMetrics(args.metricsName)
    features = []
    with torch.no_grad():
        for batch_data in tqdm(dataloader[mode]):
            data = batch_data['data'].to(args.device)
            labels = batch_data['labels'].to(args.device)
            emotions = batch_data['emotions']
            # model
            output  = model(data)
            features.append(output['fer_feature'].cpu().numpy())
            y_true.append(labels.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(y_true, axis=0)
    return features, labels

def run(args):
    if not os.path.exists(args.res_save_path):
        os.mkdir(args.res_save_path)
    # get dst dataset params
    config = Config(args)
    args = config.get_config()
    # train_df0 = pd.read_csv(os.path.join(args.label_dir,'train.csv'))
    # kf = KFold(10,shuffle = False)
    # for knum, indexs in enumerate(kf.split(train_df0)):
    #     # print(indexs)
    #     args.train_df = train_df0.iloc[indexs[0],:]
    #     args.test_df = train_df0.iloc[indexs[1],:]
    #     break
    args.train_df = pd.read_csv(os.path.join(args.label_dir,'train.csv'))
    args.test_df = pd.read_csv(os.path.join(args.label_dir,'test.csv'))
    # get dataloader
    dataloader = FERDataLoader(args)
    # gpu
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.gpu_ids[0] if using_cuda else 'cpu')
    args.device = device
    # build model
    model = AMIO(args).to(device)
    atio = ATIO().getTrain(args)
    # load best model
    model_save_pathes = glob(os.path.join(args.model_save_path,\
                                            f'{args.modelName}-{args.datasetName}.pth'))
    assert len(model_save_pathes) == 1
    model.load_state_dict(torch.load(model_save_pathes[0]))
    # do test
    mode = 'test'
    features, labels = do_test(args, model, dataloader, mode=mode)
    save_path = os.path.join(args.res_save_path, f'{args.modelName}-{args.datasetName}-{mode}.npz')
    np.savez(save_path, features=features, labels=labels)

    mode = 'train'
    features, labels = do_test(args, model, dataloader, mode=mode)
    save_path = os.path.join(args.res_save_path, f'{args.modelName}-{args.datasetName}-{mode}.npz')
    np.savez(save_path, features=features, labels=labels)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_mode', type=bool, default=False,
                        help='is transfer test ?')
    parser.add_argument('--val_mode', type=bool, default=False,
                        help='10 folds cross validation ?')
    parser.add_argument('--modelName', type=str, default='FER_DCNN',
                        help='support FER_DCNN/Our')
    parser.add_argument('--datasetName', type=str, default='RAF',
                        help='support RAF/SFEW2/CK+/OULU_CASIA')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='results/bestModels',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results/Features',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # args.seeds = [1, 12, 123, 1234, 12345]
    run(args)
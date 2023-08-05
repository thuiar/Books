import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from utils.functions import Storage

# db interaction
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))


def Process(args):
    
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
        data_config = config['data'][args.datasetName]

    args = Storage(dict(vars(args), **data_config))

    os.system('cp ' + args.True_label + ' ' + args.label_path) 
    
    df = pd.read_csv(args.label_path, encoding='utf-8', dtype={"video_id": "str", "clip_id": "str"})
    length = len(df)
    print(length)
    select_list = random.sample([i for i in range(length)], int(args.labeled * length))

    for i in range(len(df)):
        label_by, video_id, clip_id = df.loc[i, ['label_by', 'video_id', 'clip_id']]
        if i in select_list:
            df.loc[i, ['label_by']] = 0
        else:
            df.loc[i, ['label_by']] = -1

    df.to_csv(args.label_path, index=None, encoding='utf-8')
    return

def transferProcess(args):

    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
        data_config = config['data'][args.datasetName]

    args = Storage(dict(vars(args), **data_config))

    os.system('cp ' + args.True_label + ' ' + args.label_path) 
    os.system('cp ' + args.True_label_MOSEI + ' ' + args.label_path_MOSEI) 

    df = pd.read_csv(args.label_path, encoding='utf-8', dtype={"video_id": "str", "clip_id": "str"})

    for i in range(len(df)):
        label_by, video_id, clip_id = df.loc[i, ['label_by', 'video_id', 'clip_id']]
        df.loc[i, ['label_by']] = 0

    df.to_csv(args.label_path, index=None, encoding='utf-8')

    df = pd.read_csv(args.label_path_MOSEI, encoding='utf-8', dtype={"video_id": "str", "clip_id": "str"})

    for i in range(len(df)):
        label_by, video_id, clip_id = df.loc[i, ['label_by', 'video_id', 'clip_id']]
        df.loc[i, ['label_by']] = -1

    df.to_csv(args.label_path_MOSEI, index=None, encoding='utf-8')
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', type=str, default='MOSEI',
                        help='support MOSI/MOSEI')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='human labeled')
    return parser.parse_args()

if __name__ == "__main__":
    process()
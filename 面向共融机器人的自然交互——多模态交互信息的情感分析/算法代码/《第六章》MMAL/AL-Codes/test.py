import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd

# def remove0(args):
#     os.system('cp ' + args.True_label + ' ' + args.label_path)
#     df = pd.read_csv(args.label_path, encoding='utf-8', dtype={"video_id": "str", "clip_id": "str"})
#     length = len(df)
#     print(df)
#     df.drop(df[df.label == 1].index, inplace=True)
#     print(df)
#     # for i in range(len(df)):
#     #     label_by, video_id, clip_id, label = df.loc[i, ['label_by', 'video_id', 'clip_id', 'label']]
#     #     if label == 1:
#     #         df.drop(df.index[i], inplace=True)
#     #     if label == 2:
#     #         df.loc[i, ['label']] = 1
#     df.reset_index(drop = True, inplace = True)
#     print(df)
#     # record = []
#     # for i in range(len(df)):
#     #     label_by, label = df.loc[i, ['label_by', 'label']]
#     #     if label == 2:
#     #         record.append(i)
#     # for line in record:
#     #     df.loc[line, 'label'] = 1
#     # print(df)
#     df.to_csv(args.True_label, index=None, encoding='utf-8')
#     return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--True_label', type=str, default="/home/wujiele/dataset/StandardDatasets/MOSI/label.csv",
                        help='')
    parser.add_argument('--label_path', type=str, default="/home/wujiele/dataset/StandardDatasets/MOSI/label-AL.csv",
                        help='')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # remove0(args)
    list = [1,2,3,4]
    index = np.asarray(list, dtype=np.int)
    index = index - 1
    print(index)

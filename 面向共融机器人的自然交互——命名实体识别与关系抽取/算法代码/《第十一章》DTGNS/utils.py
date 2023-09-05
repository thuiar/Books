import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import random
import pickle
import csv
import sys
import math
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch.nn.utils import weight_norm
rootPath = ""


def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
          
    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['Seen'] = f_seen
    result['Unseen'] = f_unseen
    result['Overall'] = f
        
    return result
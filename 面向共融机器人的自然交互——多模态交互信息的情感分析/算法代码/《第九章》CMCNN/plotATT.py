#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/iyuge2/Project/FER')

import os
import time
import torch
import random
import argparse
import cv2
from math import pi, sqrt
from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from math import cos,sin,pi
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict

from trains import *
from utils.log import *
from utils.metricsTop import *
from utils.functions import *
from config import *
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import FERDataLoader 

plt.rc('font',family='Times New Roman',size='13')


# In[2]:


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)


# In[3]:


args = {
    'modelName': 'FER_DCNN', # FER_DCNN/LM_DCNN/MTL_HPS/MTL_PS_MCNN/MTL_MRN/MTL_CSN
    'datasetName': 'RAF',
    'val_mode': False,
    'num_workers': 0,
}
config = Config(args)
args = config.get_config()


# In[4]:


device = torch.device('cuda:3')
dataloader = FERDataLoader(args)
model = AMIO(args).to(device)


# In[5]:


model_save_path = '/home/iyuge2/Project/FER/results/bestModels2/' + args.modelName + '-' + args.datasetName + '.pth'
model.load_state_dict(torch.load(model_save_path))


# In[6]:


y_preds, y_trues = [], []
images, atts = [], []
with torch.no_grad():
    for batch_data in tqdm(dataloader['test']):
        data = batch_data['data'].to(device)
        fer_labels = batch_data['labels'].to(device)
        lm_labels = batch_data['landmarks'].to(device)
        emotions = batch_data['emotions']
        # forward
        output = model(data)
        att, fer_features, fer_out = output['fer_att'], output['fer_feature'], output['fer_output']
        # show results
        preds = fer_out.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        labels = fer_labels.detach().cpu().numpy()
        
        img = data.detach().cpu().squeeze().numpy()
        att = att.detach().cpu().squeeze().numpy()
        
        y_preds.append(preds)
        y_trues.append(labels)
        
        images.append(img)
        atts.append(att)

y_preds = np.concatenate(y_preds, axis=0)
y_trues = np.concatenate(y_trues, axis=0)

images = np.concatenate(images, axis=0)
atts = np.concatenate(atts, axis=0)
print(images.shape, atts.shape)


# In[7]:


indexes = []
for i in range(7):
    indexes.append(np.where(y_trues == i)[0])


# In[8]:


standard_label_dict = {0: 'Happiness', 1: 'Neutral', 2: 'Sadness', 3: 'Disgust', 4: 'Fear', 5: 'Surprise', 6: 'Anger'}


# In[9]:


images = (images + 1) / 2


# In[10]:

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

aa = [[724,1959,1994,854],
      [2989,2667,2771,2458],
      [718,1469,844,809],
      [6,1005,1883,2342],
      [2294,622,2243,2245],
      [1028,1608,1868,694],
      [1738,1538,2074,1829]]
fig = plt.figure(figsize=[14,8])
for i in range(4):
    for j in range(7):
        # ii = random.choice([v for v in range(len(indexes[6]))])
        # index = indexes[j][aa[j][i]]
        index = aa[j][i]
        
        # 开始绘图
        plt.subplot(4, 7, i * 7 + j + 1)
        cur_image = np.transpose(images[index], [1,2,0])
        plt.imshow(cur_image)
        plt.axis('off')

        att = Image.fromarray(atts[index])
        att_mask = att.resize([224, 224],Image.ANTIALIAS)
        plt.imshow(np.array(att_mask), alpha=0.6, cmap='jet')
        plt.axis('off')
        
        # plt.text(0, 110, ii)
        
# plt.show()
plt.savefig('./BaseDCNN_ATT.pdf', dpi=600, format='pdf', transparent=True, pad_inches = 0)

with open('/home/iyuge2/Project/FER/results/20210731/results/RAF/FER_DCNN_loss.pkl', 'rb') as tf:
    fer_losses = pickle.load(tf)
plt.plot([i for i in range(0,len(fer_losses['Train']), 100)], fer_losses['Train'][::100])
plt.plot([i for i in range(0, len(fer_losses['Valid']), 20)], fer_losses['Valid'][::20])

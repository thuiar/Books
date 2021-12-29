# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 17:09:15 2016

@author: yuhui
"""
import numpy as np

def load_data():  
    import cPickle as pickle
    #with open('x_data_train.pkl','rb') as f:
    #    train_x=pickle.load(f)
    #with open('y_data_train.pkl','rb') as f:
    #    train_y=pickle.load(f)
    with open('x_data_test.pkl','rb') as f:
        test_x=pickle.load(f)
    with open('y_data_test.pkl','rb') as f:
        test_y=pickle.load(f)
    test_len = int(len(test_x) /2)
#    train_x = train_x / np.float32(255.0)
    test_x = test_x / np.float32(255.0)
    train_x = test_x[-test_len:]
    train_y = test_y[-test_len:]
    return train_x, train_y

import scipy.io as sio
train_x, train_y = load_data()
sio.savemat('data1.mat', {
'train_x': train_x,
'train_y': train_y
})
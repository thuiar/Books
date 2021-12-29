# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:50:55 2016

@author: yuhui
"""
import numpy as np
import scipy.io as sio
from skimage import transform, io
#import cPickle as pickle

def distance (x1, y1, x2, y2):    ##used to calculate the distance between two patch
    temp=abs(x1-x2)+abs(y1-y2)
    return temp
    
    
def sliding (img, labels):   ###sliding window
    width, height = img.shape
    patch_size = 180
    stride = 36
    map_width = int((width - patch_size) / stride + 1)   ##sliding iteration times
    map_height = int((height - patch_size) / stride + 1)
    
    print("labels number=",len(labels))


    X_negative = np.zeros((map_width, map_height, 64, 64)).astype('uint8')
    y_negative = np.zeros((map_width, map_height)).astype('uint8')
    for i in range(map_width):
        for j in range(map_height):
            patch = img[i * stride: i * stride + patch_size,
                        j * stride: j * stride + patch_size] / 255.0
            X_negative[i, j] = transform.resize(patch, (64, 64)) * 255
            x_center = i * stride + patch_size / 2
            y_center = j * stride + patch_size / 2
            dist = distance(labels[:, 0], labels[:, 1], x_center, y_center)
            decision = np.where(dist <= 36)[0]  ##threshold of the distance between labels and real point
            if (len(decision) == 0):      ##Label
                y_negative[i, j] = 0
            else:
                y_negative[i, j] = 1

    X_negative = X_negative.reshape(-1, 64, 64)
    y_negative = y_negative.reshape(-1)
    X_negative = X_negative[y_negative == 0]    ##Get negative parts
    y_negative = y_negative[y_negative == 0]
    
    print ("negative examples=", len(y_negative))
    
    X_positive = np.zeros((len(labels) * 3 * 3, 64, 64)).astype('uint8')
    y_positive = np.zeros((len(labels) * 3 * 3)).astype('uint8')
    count = 0
    for i in range(len(labels)):
        x = labels[i, 0]
        y = labels[i, 1]
        #print ("place in the image",x,y)
        for i_offset in range(-4, 5, 4):
            for j_offset in range(-4, 5, 4):
                x1 = x + i_offset - patch_size / 2
                x2 = x + i_offset + patch_size / 2
                y1 = y + j_offset - patch_size / 2
                y2 = y + j_offset + patch_size / 2
                if (x1 >= 0 and x2 <= width) and (y1 >= 0 and y2 <= height):
                    patch = img[x1:x2, y1:y2] / 255.0
                    patch = transform.resize(patch, (64, 64)) * 255
                    X_positive[count] = patch
                    y_positive[count] = 1
                count += 1

    X_positive = X_positive[y_positive == 1]  ##Get positive parts
    y_positive = y_positive[y_positive == 1]
    
    print ("positive examples=", len(y_positive))
    
    indices = np.arange(len(X_negative))   ##Do the random shuffle now
    np.random.shuffle(indices)
    X_negative = X_negative[indices]
    y_negative = y_negative[indices]

    X = np.concatenate([X_negative[:len(y_positive)], X_positive], axis=0) \
        .astype('uint8')
    y = np.concatenate([y_negative[:len(y_positive)], y_positive], axis=0) \
        .astype('uint8')
        
    print("final negative=", sum(y == 0))  ##final check
    print("final positive=", sum(y == 1))
    return (X, y)

#def rotation(img,labels):

def create_data(j):
    return_x = np.zeros((0, 64, 64)).astype('uint8')  ##main function, data processing
    return_y = np.zeros((0)).astype('uint8')
    for i in range(2,j):
        img=io.imread('processed'+str(i)+'.jpg','r')
        with open ('final'+str(i)+'.txt','r') as f:
            labels = np.zeros((len(f.readlines()),2))
        with open ('final'+str(i)+'.txt','r') as f:
            count = 0
            for line in f:
                line = line.strip('\n')
                line = line.strip()
                line = line.split(',')
                line[0] = float(line[0])
                line[1] = float(line[1])
                labels[count,:] = line
                count += 1
        #print ("labels=",labels)
        print ("count=",count)
        X, y = sliding(img, labels)
        return_x = np.concatenate([return_x, X], axis=0)
        return_y = np.concatenate([return_y, y], axis=0)     
    print("save examples=",len(return_y))
    return return_x,return_y

#sio.savemat('data.mat', {
#'train_x': train_x,
#'train_y': train_y
#})

train_x,train_y=create_data(41)
train_x = train_x.reshape(-1, 64 * 64)
#temp1 = train_x[:, 0]
#train_x[:, 0] = train_x[:, 1]
#train_x[:, 1] = temp1
#train_x = train_x.reshape(64, 64, -1)
final_y = np.zeros((len(train_y),2)).astype('uint8')
final_y[:,1] = train_y
final_y[:,0] = train_x[:,0]

sio.savemat('data.mat', {
'train_x': train_x,
'train_y': final_y
})


#with open('x_data.pkl','wb') as f:
#    pickle.dump(train_x,f)
#with open('y_data.pkl','wb')as f:
#    pickle.dump(train_y,f)

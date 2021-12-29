# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:35:44 2016

@author: yuhui
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def k_means_process(image_file):
    img = cv2.imread(image_file,0)#image read be 'gray'
    plt.subplot(221),plt.imshow(img,'gray'),plt.title('original')
    plt.xticks([]),plt.yticks([])

    #change img(2D) to 1D
    img1 = cv2.medianBlur(img,85)   #########################
    plt.subplot(222),plt.imshow(img1,'gray'),plt.title('medianBlur')
    plt.xticks([]),plt.yticks([])
    img1 = img1.reshape((img1.shape[0]*img.shape[1],1))
    img1 = np.float32(img1)
    #define criteria = (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,4.0)   #############
    
    #set flags: hou to choose the initial center
    #---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_RANDOM_CENTERS
    # apply kmenas
    compactness,labels,centers = cv2.kmeans(img1,2,criteria,20,flags)
    print ('labels0=',len(img1[labels==0]))
    print ('labels1=',len(img1[labels==1]))
    img2 = labels.reshape((img.shape[0],img.shape[1]))
    plt.subplot(223),plt.imshow(img2,'gray'),plt.title('kmeans')
    plt.xticks([]),plt.yticks([])    
    label_k = -1
    if ((len(img1[labels==0]) < len(img1[labels==1])) and (len(img1[labels==0]) > 4000) and (len(img1[labels==0]) < 900000)):
        label_k = 0
    elif ((len(img1[labels==0]) > len(img1[labels==1])) and (len(img1[labels==1]) > 4000) and (len(img1[labels==1]) < 900000)):
        label_k = 1
    
    if (label_k == -1):
        print ('wrong running')
        return 0
    
    tmp1 = img.shape[0]
    tmp2 = img.shape[1]
    img = img.reshape((tmp1 * tmp2,1))
    img = np.float32(img)
    image_normal_mean = np.mean(img[labels==(1 - label_k)])
    img[labels==label_k] *= image_normal_mean / np.mean(img[labels==label_k])
    img = img.reshape((tmp1, tmp2))
    return img

file_number=1
img_file = str(file_number)+'.jpg'
img = k_means_process(img_file)
while (not img.any()):
    img = k_means_process(img_file)
cv2.imwrite('final'+str(img_file), img)
    
    

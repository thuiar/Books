# -*- coding: utf-8 -*-
"""
Created on Tue May 31 01:21:51 2016

@author: yuhui
"""

from skimage import io, filters, exposure, img_as_float
import matplotlib.pyplot as plt
from skimage.morphology import disk
import numpy as np

def preprocessing(file_name):
    img = io.imread(file_name, as_grey = True)
    img = img_as_float(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    gaussian_img = filters.gaussian_filter(img, sigma=2.5)
    median_img = filters.median(gaussian_img,disk(5))
    
    hist_img=exposure.equalize_hist(median_img)
    median_img2 = filters.median(hist_img,disk(5))
    plt.subplot(221),plt.imshow(img,'gray'),plt.title('original')
    plt.xticks([]),plt.yticks([])
    
    plt.subplot(222),plt.imshow(gaussian_img,'gray'),plt.title('gaussian_filter')
    plt.xticks([]),plt.yticks([])
    
    plt.subplot(223),plt.imshow(median_img,'gray'),plt.title('median_filter')
    plt.xticks([]),plt.yticks([])
    
    plt.subplot(224),plt.imshow(hist_img,'gray'),plt.title('final_hist')
    plt.xticks([]),plt.yticks([])

    plt.show()
    
    return median_img2


file_name = '4.jpg'
img = preprocessing(file_name)
#for i in [3, 6, 11, 14, 15, 17, 18, 22, 29, 36, 39]:
#    file_name = 'final'+str(i)+'.jpg'
#    img = preprocessing(file_name)
#    io.imsave('processed'+str(i)+'.jpg',img)

#for i in ([2]+range(4,6)+range(7,11)+range(12,14)+[16]+range(19,22)+range(23,29)+range(30,36)+range(37,39)+[40]):
#    file_name=str(i)+'.jpg'
#    img=preprocessing(file_name)
#    io.imsave('processed'+str(i)+'.jpg',img)
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:03:40 2016

@author: yuhui
"""

import matplotlib.pyplot as plt
import numpy as np
#from skimage.filters.rank import entropy
#from skimage.morphology import disk
from skimage import io, img_as_float
#from skimage.exposure import rescale_intensity
from skimage.morphology import reconstruction
#from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
#from math import sqrt
#from skimage.filters import threshold_otsu
#from skimage.segmentation import random_walker
from scipy.ndimage import gaussian_filter

def gaussian_mask(file_number):
    img_file = str(file_number)+'.jpg'
    img=io.imread(img_file)
    image=rgb2gray(img)
#entr_img = entropy(img, disk(25))
#image=rescale_intensity(img, in_range=(50, 200))
#thresh = threshold_otsu(image)
#binary = image > thresh
#markers = np.zeros(image.shape, dtype=np.uint)
#markers[image < thresh] = 1
#markers[image > thresh] = 2
#labels = random_walker(image, markers, beta=1, mode='bf')

    image = gaussian_filter(image, 35)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')

#seed = np.copy(image)￼
#seed[1:-1, 1:-1] = image.max()
#mask = image

#filled = reconstruction(seed, mask, method='erosion')
#result=mask-filled

#blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    result=np.zeros(np.shape(img))
    result[np.where(dilated<np.mean(dilated)*0.92)]=1
    
    img = img_as_float(img)
    img_normal_mean = np.mean(img[np.where(dilated>=np.mean(dilated)*0.92)])
    return_image = np.copy(img)
    return_image[np.where(dilated<np.mean(dilated)*0.92)] = img_normal_mean #/ np.mean(img[np.where(dilated<np.mean(dilated)*0.95)])


    plt.subplot(221),plt.imshow(img,'gray'),plt.title('original')
    plt.xticks([]),plt.yticks([])

    plt.subplot(222),plt.imshow(dilated,'gray'),plt.title('bubble_detected')
    plt.xticks([]),plt.yticks([])
    
    plt.subplot(223),plt.imshow(result,'gray'),plt.title('bubble_binary')
    plt.xticks([]),plt.yticks([])
    
    plt.subplot(224),plt.imshow(return_image,'gray'),plt.title('final_image')
    plt.xticks([]),plt.yticks([])

    plt.show()

    return return_image

###3
#for i in range(0,41):
#    img=gaussian_mask(i)
#    io.imsave('final'+str(i)+'.jpg',img)
#    print('save completed, file number is ', i)

###6
#for i in range(0,21):
#    img=gaussian_mask(i)
#    io.imsave('final'+str(i)+'.jpg',img)
#    print('save completed, file number is ', i)

i = 3
img = gaussian_mask(i)
io.imsave('final'+str(i)+'.jpg',img)
print ('save completed')
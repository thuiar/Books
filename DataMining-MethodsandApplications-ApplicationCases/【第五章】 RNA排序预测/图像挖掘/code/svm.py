# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 10:54:00 2016

@author: yuhui
"""
import numpy as np
import time

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
    train_len = int(len(test_x) / 4)
    test_len = int(len(test_x) /2)
#    train_x = train_x / np.float32(255.0)
    test_x = test_x / np.float32(255.0)
    train_x, test_x = test_x[-test_len:], test_x[:train_len]
    train_y, test_y = test_y[-test_len:], test_y[:train_len]    
    return train_x, train_y, test_x, test_y

from sklearn import svm
from sklearn import metrics  


print 'reading training and testing data...'  
train_x, train_y, test_x, test_y = load_data()  
num_train, num_feat = train_x.shape  
num_test, num_feat = test_x.shape
print '******************** Data Info *********************'  
print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)

print '******************* svm ********************' 
start_time = time.time()  
clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto')
clf.fit(train_x, train_y)
print 'training took %fs!' % (time.time() - start_time)  


train_result=clf.predict(train_x)
predict = clf.predict(test_x) 

precision = metrics.precision_score(train_y, train_result)  
recall = metrics.recall_score(train_y, train_result)  
print 'train precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)  
accuracy = metrics.accuracy_score(train_y, train_result)  
print 'train accuracy: %.2f%%' % (100 * accuracy)

precision = metrics.precision_score(test_y, predict)  
recall = metrics.recall_score(test_y, predict)  
print 'test precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)  
accuracy = metrics.accuracy_score(test_y, predict)  
print 'test accuracy: %.2f%%' % (100 * accuracy)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

test_dec = clf.decision_function(test_x)
fpr, tpr, _ = roc_curve(test_y, test_dec)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# -*- coding: utf-8 -*-
from sklearn import svm
import cPickle
import numpy as np

def detail_result(predict,label,classnum):
    assert len(predict)==len(label)
    results = np.zeros((classnum),np.int32)
    lengths = np.zeros((classnum),np.int32)
    for i in range(classnum):
        indices = label==i
        sublabel = label[indices]
        subpredict = predict[indices]
        results[i] = sum(sublabel==subpredict)
        lengths[i] = len(sublabel)
    return results,lengths
filename="suboptTrain"
with file(filename,'rb') as f:
    gene = cPickle.load(f)
    label = cPickle.load(f)
filename="suboptTest"
with file(filename,'rb') as f:
    test_gene = cPickle.load(f)
    test_label = cPickle.load(f)
#%%

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
scaler = preprocessing.StandardScaler().fit(gene.astype(np.float))
gene = scaler.transform(gene.astype(np.float))
test_gene = scaler.transform(test_gene.astype(np.float))
#%%
"""
temp = np.reshape(gene,(gene.shape[0],gene.shape[2],gene.shape[3]))
temp = np.transpose(temp,(0,2,1))
x = np.reshape(temp,(temp.shape[0],temp.shape[1]*temp.shape[2]))
"""
#%%
clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto', cache_size=2000, class_weight='balanced')
clf.fit(gene, label)

train_result=clf.predict(gene)
train_acc,train_len=detail_result(train_result,label,2)
test_result=clf.predict(test_gene)
test_acc,test_len=detail_result(test_result,test_label,2)
print "train TP=",float(train_acc[1])/train_len[1],"TN=",float(train_acc[0])/train_len[0]
print "test TP=",float(test_acc[1])/test_len[1],"TN=",float(test_acc[0])/test_len[0]
#%%
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
test_dec = clf.decision_function(test_gene)
fpr, tpr, _ = roc_curve(test_label, test_dec)
roc_auc = auc(fpr, tpr)
print 'roc_auc =',roc_auc
'''
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
'''

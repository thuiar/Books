# -*- coding: utf-8 -*-
from sklearn import svm
import cPickle
import numpy as np
from sklearn.decomposition import PCA
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
# store limited dimensions

gene_first = gene_first_save = gene[:, 0:132]
gene_second = gene_second_save = gene[:, 132:]
del gene

#print gene_first.shape
#print gene_second.shape

filename="suboptTest"
with file(filename,'rb') as f:
    test_gene = cPickle.load(f)
    test_label = cPickle.load(f)


test_first = test_first_save = test_gene[:, 0:132]
test_second = test_second_save = test_gene[:, 132:]

del test_gene
#print test_first.shape
#print test_second.shape
#print test_second2.shape
#print test_second3.shape

save_file = open('save_pca_svm1.txt','w')
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

for first_i in range(1, 21):
    for second_j in range(1, 21):
        pca_first_dim = first_i
        pca_second_dim = second_j

        print 'PCA dimension for first strucutre =',pca_first_dim
        print 'PCA dimension for second structure =',pca_second_dim
        
        gene_first = gene_first_save
        gene_second = gene_second_save
                
        test_first = test_first_save
        test_second = test_second_save
                
        pca_first = PCA(n_components = pca_first_dim)
        pca_second = PCA(n_components = pca_second_dim)
        
        pca_first.fit(gene_first)
        pca_second.fit(gene_second)

                
        gene_first=pca_first.transform(gene_first)
        gene_second=pca_second.transform(gene_second)
                
        gene=np.concatenate((gene_first,gene_second),axis=1)
        test_first=pca_first.transform(test_first)
        test_second=pca_second.transform(test_second)

        test_gene=np.concatenate((test_first,test_second),axis=1)
        
        
        scaler = preprocessing.StandardScaler().fit(gene.astype(np.float))
        gene = scaler.transform(gene.astype(np.float))
        test_gene = scaler.transform(test_gene.astype(np.float))
        
        clf = svm.SVC(C=2, kernel='rbf', degree=3, gamma='auto', cache_size=2000, class_weight='balanced')
        clf.fit(gene, label)
        
        train_result=clf.predict(gene)
        train_acc,train_len=detail_result(train_result,label,2)
        test_result=clf.predict(test_gene)
        test_acc,test_len=detail_result(test_result,test_label,2)
            
        train_TP=float(train_acc[1])/train_len[1]
        train_TN=float(train_acc[0])/train_len[0]
        test_TP=float(test_acc[1])/test_len[1]
        test_TN=float(test_acc[0])/test_len[0]
            
        print "train TP=",float(train_acc[1])/train_len[1],"TN=",float(train_acc[0])/train_len[0]
        print "test TP=",float(test_acc[1])/test_len[1],"TN=",float(test_acc[0])/test_len[0]
#%%
#        import matplotlib.pyplot as plt
        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc
        from sklearn.cross_validation import train_test_split
        from sklearn.preprocessing import label_binarize
        from sklearn.multiclass import OneVsRestClassifier
        test_dec = clf.decision_function(test_gene)
        fpr, tpr, _ = roc_curve(test_label, test_dec)
        roc_auc = auc(fpr, tpr)
        print 'roc_auc =',roc_auc
        if (train_TP>0.6 and train_TN>0.6 and test_TP>0.6 and test_TN>0.6):
            save_file.write('*****svm****round%d***************' %((first_i-1)*20 + second_j))
            save_file.write('\n')
            save_file.write('PCA dimension for first structure = %d' %(pca_first_dim))
            save_file.write('\n')
            save_file.write('PCA dimension for second structure = %d' %(pca_second_dim))
            save_file.write('train_TP = %f' %train_TP)
            save_file.write('\n')
            save_file.write('train_TN = %f' %train_TN)
            save_file.write('\n')
            save_file.write('test_TP = %f' %test_TP)
            save_file.write('\n')
            save_file.write('test_TN = %f' %test_TN)
            save_file.write('\n')
            save_file.write('roc_auc = %f' %roc_auc)
            save_file.write('\n')
        print 'svm round %d is completed' %((first_i-1)*20+second_j)

save_file.close()

'''
pca_dim=5
print 'PCA dimension =',pca_dim
pca = PCA(n_components=pca_dim)
pca.fit(gene)
gene=pca.transform(gene)
filename="4merTest"
with file(filename,'rb') as f:
    test_gene = cPickle.load(f)
    test_label = cPickle.load(f)
test_gene=pca.transform(test_gene)
#%%
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
scaler = preprocessing.StandardScaler().fit(gene.astype(np.float))
gene = scaler.transform(gene.astype(np.float))
test_gene = scaler.transform(test_gene.astype(np.float))
#%%
'''
'''
temp = np.reshape(gene,(gene.shape[0],gene.shape[2],gene.shape[3]))
temp = np.transpose(temp,(0,2,1))
x = np.reshape(temp,(temp.shape[0],temp.shape[1]*temp.shape[2]))
'''
'''
#%%
clf = svm.SVC(C=2, kernel='rbf', degree=3, gamma='auto', cache_size=2000, class_weight='balanced')
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

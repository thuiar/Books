# coding: utf-8
'''
X. Jin
'''
import numpy as np
import os
import cPickle
import time
import random
import math
import ctypes as ct
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pdb
#dll=np.ctypeslib.load_library('deepboost.so','.')
seq_len = 174


def stride_kmer(seq, k, stride,lib):
    """
    seq is a sequence str
    stride could be designed
    stride_kmer calculate count vector of seq_i seq_{i+stride} seq_{i+2stride} ... seq_{i+(k-1)stride}
    """
    for i in range(len(seq)-stride*(k-1)):
        temp=seq[i]
        for j in range(k-1):
            temp=temp+seq[i+(j+1)*stride]
        if not lib.has_key(temp):
            lib[temp] = 1
        else:
            lib[temp] += 1
    return lib

def stride_count_vector(seq, k, stride):
    """
    return the float32 count vector of k-mer
    size: np.power(4,k)
    """
    kmerlib={}
    # count k-mer
    kmerlib=stride_kmer(seq,k,stride,kmerlib)
    result = np.zeros((np.power(4,k),),np.float32)
    index = np.linspace(0,k-1,num=k,dtype=int)
    index = np.power(4,index)
    for line in kmerlib:
        #print line
        a = np.firstarray(list(line),dtype=int)-np.ones((len(line),),dtype=int)
        result[np.dot(a,index)] = kmerlib[line]
    result = result / np.sum(result)
    return result
"""
eglib={}
seq="123412341234"
print stride_count_vector(seq,2,4)
"""


# In[4]:

def kmer_number(seq,k,lib):
    """
    seq is a sequence str
    k = k of k-mer
    lib is a dictionary of [kmer, count of kmer] pair
    """
    for i in range(len(seq)-k+1):
        if not lib.has_key(seq[i:i+k]):
            lib[seq[i:i+k]] = 1
        else:
            lib[seq[i:i+k]] += 1
    return lib

def count_vector(seq,k):
    """
    return the float32 count vector of k-mer
    size: np.power(4,k)
    """
    kmerlib={}
    # count k-mer
    kmerlib=kmer_number(seq,k,kmerlib)
    result = np.zeros((np.power(4,k),),np.float32)
    index = np.linspace(0,k-1,num=k,dtype=int)
    index = np.power(4,index)
    for line in kmerlib:
        #print line
        a = np.firstarray(list(line),dtype=int)-np.ones((len(line),),dtype=int)
        result[np.dot(a,index)] = kmerlib[line]
    result = result / np.sum(result)
    return result

def one_hot_encode(seqs,cls):
    num,length=np.shape(seqs)
    result=np.zeros((num,cls,length),np.float32)
    for i in xrange(num):
        for j in xrange(length):
            result[i,seqs[i,j],j] = 1
    return result

# In[11]:

def distinct_kmer_number(seq,k,lib,otherlib):
    """
    seq is a sequence str
    k = k of k-mer
    lib is a dictionary of [kmer, count of kmer] pair and exclude those in otherlib
    """
    for i in range(len(seq)-k+1):
        if not otherlib.has_key(seq[i:i+k]):
            if not lib.has_key(seq[i:i+k]) :
                lib[seq[i:i+k]] = 1
            else:
                lib[seq[i:i+k]] += 1
    return lib
"""
#example
print "example:"
eglib={}
egseq="123412341234"
egneglib={"1234":1,"1324":1,"4123":1}
print "neg lib=",egneglib
print "4mer only in egneglib=",distinct_kmer_number(egseq,4,eglib,egneglib)
"""
# calculate features
"""
k=1,2,3
k=2 + stride=4,5,6
"""
# input data
with open('pos_subopt_data','rb') as f:
    plib = cPickle.load(f)
    psecond1 = cPickle.load(f)
    psecond2 = cPickle.load(f)
    psecond3 = cPickle.load(f)
with open('neg_subopt_data','rb') as f:
    nlib = cPickle.load(f)
    nsecond1 = cPickle.load(f)
    nsecond2 = cPickle.load(f)
    nsecond3 = cPickle.load(f)
plen = len(plib)
nlen = len(nlib)
pfirst = np.array(plib,np.int32)-np.ones((len(plib),len(plib[0])),np.int32)
pfirst = one_hot_encode(pfirst,4)
nfirst = np.array(nlib,np.int32)-np.ones((len(nlib),len(nlib[0])),np.int32)
nfirst = one_hot_encode(nfirst,4) 

pfirst= np.expand_dims(pfirst,axis=1)
nfirst = np.expand_dims(nfirst,axis=1)

cls = 5
pcode1 = one_hot_encode(psecond1,cls)
pcode2 = one_hot_encode(psecond2,cls)
pcode3 = one_hot_encode(psecond3,cls)
ncode1 = one_hot_encode(nsecond1,cls)
ncode2 = one_hot_encode(nsecond2,cls)
ncode3 = one_hot_encode(nsecond3,cls)

pcode1 = np.expand_dims(pcode1,axis=1)
pcode2 = np.expand_dims(pcode2,axis=1)
pcode3 = np.expand_dims(pcode3,axis=1)
ncode1 = np.expand_dims(ncode1,axis=1)
ncode2 = np.expand_dims(ncode2,axis=1)
ncode3 = np.expand_dims(ncode3,axis=1)

psecond = np.concatenate((pcode1,pcode2,pcode3),axis=1)
nsecond = np.concatenate((ncode1,ncode2,ncode3),axis=1)
print 'distinct # of positive sample =',plen
print 'distinct # of negative sample =',nlen

#pdb.set_trace()

# shuffle
pindex = range(plen)
random.shuffle(pindex)
nindex = range(nlen)
random.shuffle(nindex)

neg_pos_ratio = int(math.floor(nlen/plen))

#%% cross validation
cross_valid_num = 10

pclassnum = int(math.floor(plen*0.8))
nclassnum = int(math.floor(nlen*0.8))
classnum = pclassnum+nclassnum

plastclassnum = plen - pclassnum
nlastclassnum = nlen -nclassnum
lastclassnum = plastclassnum+nlastclassnum

firstarray = np.zeros((classnum,np.shape(pfirst)[1],np.shape(pfirst)[2],np.shape(pfirst)[3]),dtype=np.float32)
secondarray = np.zeros((classnum,np.shape(psecond)[1],np.shape(psecond)[2],np.shape(psecond)[3]),dtype=np.float32)
nntarget = np.zeros((classnum,),dtype=np.int32)

lastfirstarray = np.zeros((lastclassnum,np.shape(pfirst)[1],np.shape(pfirst)[2],np.shape(pfirst)[3]),dtype=np.float32)
lastsecondarray = np.zeros((lastclassnum,np.shape(psecond)[1],np.shape(psecond)[2],np.shape(psecond)[3]),dtype=np.float32)
nnlasttarget = np.zeros((lastclassnum,),dtype=np.int32)

pj = 0
nj = 0
for i in range(classnum):
    if i%(neg_pos_ratio+1) == 0:
        firstarray[i,:,:,:] = pfirst[pindex[pj],:,:,:]
        secondarray[i,:,:,:] = psecond[pindex[pj],:,:,:]
        nntarget[i] = 1
        pj += 1
    else:
        firstarray[i,:,:,:] = nfirst[nindex[nj],:,:,:]
        secondarray[i,:,:,:] = nsecond[nindex[nj],:,:,:]
        nj += 1
        nntarget[i] = 0
pj = 0
nj = 0
for i in range(lastclassnum):
    if i%(neg_pos_ratio+1) == 0:
        if pj < plastclassnum:
            lastfirstarray[i,:,:,:] = pfirst[pindex[pj+pclassnum],:,:,:]
            lastsecondarray[i,:,:,:] = psecond[pindex[pj+pclassnum],:,:,:]
            nnlasttarget[i] = 1
            pj += 1
        else:
            lastfirstarray[i,:,:,:] = nfirst[nindex[nj+nclassnum],:,:,:]
            lastsecondarray[i,:,:,:] = nsecond[nindex[nj+nclassnum],:,:,:]
            nj += 1
            nnlasttarget[i] = 0
    else:
        if nj < nlastclassnum:
            lastfirstarray[i,:,:,:] = nfirst[nindex[nj+nclassnum],:,:,:]
            lastsecondarray[i,:,:,:] = nsecond[nindex[nj+nclassnum],:,:,:]
            nj += 1
            nnlasttarget[i] = 0
        else:
            lastfirstarray[i,:,:,:] = pfirst[pindex[pj+pclassnum],:,:,:]
            lastsecondarray[i,:,:,:] = psecond[pindex[pj+pclassnum],:,:,:]
            pj += 1
            nnlasttarget[i] = 1
'''
store data

scaler = preprocessing.StandardScaler().fit(firstarray.astype(np.float))
firstarray = scaler.transform(firstarray.astype(np.float))
lastfirstarray = scaler.transform(lastfirstarray.astype(np.float))
scaler = preprocessing.StandardScaler().fit(secondarray.astype(np.float))
secondarray = scaler.transform(secondarray.astype(np.float))
lastsecondarray = scaler.transform(lastsecondarray.astype(np.float))
'''
with open("cnn2inputTrain","wb") as f:
    cPickle.dump(firstarray,f)
    cPickle.dump(secondarray,f)
    cPickle.dump(nntarget,f)
with open("cnn2inputTest","wb") as f:
    cPickle.dump(lastfirstarray,f)
    cPickle.dump(lastsecondarray,f)
    cPickle.dump(nnlasttarget,f)



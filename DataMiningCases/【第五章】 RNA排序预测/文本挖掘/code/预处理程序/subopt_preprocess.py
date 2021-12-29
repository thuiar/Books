
# coding: utf-8

# In[2]:

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
'''
pfile = 'positive.txt'
nfile = 'neg.txt'

pstr=[]
nstr=[]
with open(pfile,'r') as pf:
    for line in pf:

        pstr.append(line[0:seq_len])

with open(nfile,'r') as nf:
    for line in nf:
        nstr.append(line[0:seq_len])

def distinct_seq(strings,lib):
    for i in range(len(strings)):
        if not lib.has_key(strings[i]):
            lib[strings[i]] = 1
        else:
            lib[strings[i]] += 1
    return lib

plib = {}
nlib = {}
plib = distinct_seq(pstr,plib)
nlib = distinct_seq(nstr,nlib)
'''


# In[3]:

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
        a = np.array(list(line),dtype=int)-np.ones((len(line),),dtype=int)
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
        a = np.array(list(line),dtype=int)-np.ones((len(line),),dtype=int)
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
cls = 5
pcode1 = one_hot_encode(psecond1,cls)
pcode2 = one_hot_encode(psecond2,cls)
pcode3 = one_hot_encode(psecond3,cls)
ncode1 = one_hot_encode(nsecond1,cls)
ncode2 = one_hot_encode(nsecond2,cls)
ncode3 = one_hot_encode(nsecond3,cls)

pcode1 = np.reshape(pcode1,(np.shape(pcode1)[0],np.shape(pcode1)[1]*np.shape(pcode1)[2]))
pcode2 = np.reshape(pcode2,(np.shape(pcode2)[0],np.shape(pcode2)[1]*np.shape(pcode2)[2]))
pcode3 = np.reshape(pcode3,(np.shape(pcode3)[0],np.shape(pcode3)[1]*np.shape(pcode3)[2]))
ncode1 = np.reshape(ncode1,(np.shape(ncode1)[0],np.shape(ncode1)[1]*np.shape(ncode1)[2]))
ncode2 = np.reshape(ncode2,(np.shape(ncode2)[0],np.shape(ncode2)[1]*np.shape(ncode2)[2]))
ncode3 = np.reshape(ncode3,(np.shape(ncode3)[0],np.shape(ncode3)[1]*np.shape(ncode3)[2]))
pcode = np.concatenate((pcode1,pcode2,pcode3),axis=1)
ncode = np.concatenate((ncode1,ncode2,ncode3),axis=1)
print 'distinct # of positive sample =',plen
print 'distinct # of negative sample =',nlen

#pdb.set_trace()
presult=np.zeros((plen,132),np.float32)
i=0
for line in plib:
    tmp=''.join(map(str, line))
    presult[i,0:4]=count_vector(tmp,1)
    presult[i,4:20]=count_vector(tmp,2)
    presult[i,20:84]=count_vector(tmp,3)
    presult[i,84:100]=stride_count_vector(tmp,2,4)
    presult[i,100:116]=stride_count_vector(tmp,2,5)
    presult[i,116:132]=stride_count_vector(tmp,2,6)
    i += 1
presult = np.concatenate((presult,pcode),axis=1)
#presult=pcode
nresult=np.zeros((nlen,132),np.float32)
i=0
for line in nlib:
    tmp=''.join(map(str, line))
    nresult[i,0:4]=count_vector(tmp,1)
    nresult[i,4:20]=count_vector(tmp,2)
    nresult[i,20:84]=count_vector(tmp,3)
    nresult[i,84:100]=stride_count_vector(tmp,2,4)
    nresult[i,100:116]=stride_count_vector(tmp,2,5)
    nresult[i,116:132]=stride_count_vector(tmp,2,6)
    i += 1
nresult = np.concatenate((nresult,ncode),axis=1)
#nresult=ncode
# In[15]:
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
"""
array = np.zeros((classnum,np.power(4,k)),dtype=np.float32)
target = np.zeros((classnum,),dtype=np.int32)
nntarget = np.zeros((classnum,),dtype=np.int32)

lastarray = np.zeros((lastclassnum,np.power(4,k)),dtype=np.float32)
lasttarget = np.zeros((lastclassnum,),dtype=np.int32)
nnlasttarget = np.zeros((lastclassnum,),dtype=np.int32)
"""
array = np.zeros((classnum,np.shape(presult)[1]),dtype=np.float32)
target = np.zeros((classnum,),dtype=np.int32)
nntarget = np.zeros((classnum,),dtype=np.int32)

lastarray = np.zeros((lastclassnum,np.shape(nresult)[1]),dtype=np.float32)
lasttarget = np.zeros((lastclassnum,),dtype=np.int32)
nnlasttarget = np.zeros((lastclassnum,),dtype=np.int32)

pj = 0
nj = 0
for i in range(classnum):
    if i%(neg_pos_ratio+1) == 0:
        array[i,:] = presult[pindex[pj],:]
        target[i] = 1
        nntarget[i] = 1
        pj += 1
    else:
        array[i,:] = nresult[nindex[nj],:]
        nj += 1
        target[i] = -1
        nntarget[i] = 0
pj = 0
nj = 0
for i in range(lastclassnum):
    if i%(neg_pos_ratio+1) == 0:
        if pj < plastclassnum:
            lastarray[i,:] = presult[pindex[pj+pclassnum],:]
            lasttarget[i] = 1 
            nnlasttarget[i] = 1
            pj += 1
        else:
            lastarray[i,:] = nresult[nindex[nj+nclassnum],:]
            nj += 1
            lasttarget[i] = -1
            nnlasttarget[i] = 0
    else:
        if nj < nlastclassnum:
            lastarray[i,:] = nresult[nindex[nj+nclassnum],:]
            nj += 1
            lasttarget[i] = -1
            nnlasttarget[i] = 0
        else:
            lastarray[i,:] = presult[pindex[pj+pclassnum],:]
            pj += 1
            lasttarget[i] = 1
            nnlasttarget[i] = 1
"""
store data
scaler = preprocessing.StandardScaler().fit(array.astype(np.float))
array = scaler.transform(array.astype(np.float))
lastarray = scaler.transform(lastarray.astype(np.float))
"""
with open("suboptTrain","wb") as f:
    cPickle.dump(array,f)
    cPickle.dump(nntarget,f)
with open("suboptTest","wb") as f:
    cPickle.dump(lastarray,f)
    cPickle.dump(nnlasttarget,f)
"""
with open("strideTrain","wb") as f:
    cPickle.dump(array,f)
    cPickle.dump(nntarget,f)
with open("strideTest","wb") as f:
    cPickle.dump(lastarray,f)
    cPickle.dump(nnlasttarget,f)
"""


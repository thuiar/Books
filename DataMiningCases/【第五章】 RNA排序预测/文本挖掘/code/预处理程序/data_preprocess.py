
# coding: utf-8

# In[2]:

import numpy as np
import os
import pickle
import time
import random
import math
import ctypes as ct
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
#dll=np.ctypeslib.load_library('deepboost.so','.')

os.system('type negative.txt unknown.txt > neg.txt') ##type=cat if linux

seq_len = 174

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
        if strings[i] not in lib:
            lib[strings[i]] = 1
        else:
            lib[strings[i]] += 1
    return lib

plib = {}
nlib = {}
plib = distinct_seq(pstr,plib)
nlib = distinct_seq(nstr,nlib)

plen = len(plib)
nlen = len(nlib)
print ('distinct # of positive sample =',plen)
print ('distinct # of negative sample =',nlen)


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
        if temp not in lib:
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
        if not seq[i:i+k] not in lib:
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


# In[11]:

def distinct_kmer_number(seq,k,lib,otherlib):
    """
    seq is a sequence str
    k = k of k-mer
    lib is a dictionary of [kmer, count of kmer] pair and exclude those in otherlib
    """
    for i in range(len(seq)-k+1):
        if seq[i:i+k] not in otherlib:
            if seq[i:i+k] not in lib :
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


k=6

neglib={}
for line in nlib:
    neglib=kmer_number(line,k,neglib)
print (len(neglib))
poslib={}
for line in plib:
    poslib=kmer_number(line,k,poslib)
print (len(poslib))
posonlylib={}
for line in plib:
    posonlylib=distinct_kmer_number(line,k,posonlylib,neglib)
print (len(posonlylib))

# In[14]:

"""
# calculate features
k=6
presult=np.zeros((plen,np.power(4,k)),np.float32)
i=0
for line in plib:
    presult[i,:] = count_vector(line,k)
    i += 1
nresult=np.zeros((nlen,np.power(4,k)),np.float32)
i=0
for line in nlib:
    nresult[i,:] = count_vector(line,k)
    i += 1
"""

# calculate features
"""
k=1,2,3
k=2 + stride=4,5,6
"""
presult=np.zeros((plen,132),np.float32)
i=0
for line in plib:
    presult[i,0:4]=count_vector(line,1)
    presult[i,4:20]=count_vector(line,2)
    presult[i,20:84]=count_vector(line,3)
    presult[i,84:100]=stride_count_vector(line,2,4)
    presult[i,100:116]=stride_count_vector(line,2,5)
    presult[i,116:132]=stride_count_vector(line,2,6)
    i += 1
nresult=np.zeros((nlen,132),np.float32)
i=0
for line in nlib:
    nresult[i,0:4]=count_vector(line,1)
    nresult[i,4:20]=count_vector(line,2)
    nresult[i,20:84]=count_vector(line,3)
    nresult[i,84:100]=stride_count_vector(line,2,4)
    nresult[i,100:116]=stride_count_vector(line,2,5)
    nresult[i,116:132]=stride_count_vector(line,2,6)
    i += 1


# In[15]:

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
array = np.zeros((classnum,132),dtype=np.float32)
target = np.zeros((classnum,),dtype=np.int32)
nntarget = np.zeros((classnum,),dtype=np.int32)

lastarray = np.zeros((lastclassnum,132),dtype=np.float32)
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

with open(str(k)+"merTrain","wb") as f:
    cPickle.dump(array,f)
    cPickle.dump(nntarget,f)
with open(str(k)+"merTest","wb") as f:
    cPickle.dump(lastarray,f)
    cPickle.dump(nnlasttarget,f)
with open(str(k)+"merTrainDpBoost","wb") as f:
    cPickle.dump(array,f)
    cPickle.dump(target,f)
with open(str(k)+"merTestDpBoost","wb") as f:
    cPickle.dump(lastarray,f)
    cPickle.dump(lasttarget,f)
"""
with open("strideTrain","wb") as f:
    pickle.dump(array,f)
    pickle.dump(nntarget,f)
with open("strideTest","wb") as f:
    pickle.dump(lastarray,f)
    pickle.dump(nnlasttarget,f)


while 1:
	pass

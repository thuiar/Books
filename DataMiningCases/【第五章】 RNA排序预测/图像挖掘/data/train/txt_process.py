# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:25:11 2016

@author: yuhui
"""

for i in range(6,10):
    save=[]
    with open('stack_010'+str(i)+'_lowpass_2x_SumCorr_manual_lgc.star',"r")as f:
        for j in range (0,9):
            f.readline()
        for line in f:
            line=line.strip('\n')
            line=line.strip()
            line=line.split()[0:2]
            line=','.join(line)
            save.append(line)
    pf=open("final"+str(i)+".txt","w")
    pf.truncate()
    for j in save:
        pf.write(j)
        pf.write('\n')
    pf.close()
            
for i in range(10,42):
    save=[]
    with open('stack_01'+str(i)+'_lowpass_2x_SumCorr_manual_lgc.star',"r")as f:
        for j in range (0,9):
            f.readline()
        for line in f:
            line=line.strip('\n')
            line=line.strip()
            line=line.split()[0:2]
            line=','.join(line)
            save.append(line)
    pf=open("final"+str(i)+".txt","w")
    pf.truncate()
    for j in save:
        pf.write(j)
        pf.write('\n')
    pf.close()            
            
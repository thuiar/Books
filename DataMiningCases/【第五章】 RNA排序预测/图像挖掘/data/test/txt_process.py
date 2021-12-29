# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:25:11 2016

@author: yuhui
"""

for i in range(1,21):
    save=[]
    with open(str(i)+".txt","r")as f:
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
            
            
            
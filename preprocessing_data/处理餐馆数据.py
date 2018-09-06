# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:44:56 2018

@author: Yingpeng_Du
"""
import pandas as pd 
import os
import numpy as np
path = '../data/rawdata/entree_data/entree/session'
A= os.listdir(path)
res = []
user = 0
n=5
l=5
for file in A:
    f = open(path+'\\'+file)
    line = f.readline()
    while line:
        if line =='' and line[-3:-1]=='-1':
            line = f.readline()
        else:            
            line = line.split('\t') 
            item = line[-1][:-1] 
            if line[2]=='0':
                feature = line[3:-1]
            else:
                feature = line[2:-1]
            feature1 = list(map(lambda x :'T'+str(x[:-1]),feature)) #time
            feature2 = list(map(lambda x :'S'+str(x[-1:]),feature)) #state
            if len(feature1)>n:                
                state_feature = feature2[-n:]#+ [feature2[-1] for i in range(n-len(feature1))]
                time_feature = feature1[len(feature1)-l:]#+ [feature1[-1] for i in range(n-len(feature1))]
                res.append([1]+['U'+str(line[1]),'I'+str(item)]+state_feature+time_feature)
                user = user + 1
            
        line = f.readline()
    f.close()
ratings = pd.DataFrame(res)
ratings.to_csv('../data'+'/positive/resturant/resturant.libfm',index=False,header=None,sep=' ')     

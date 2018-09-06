# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:18:50 2018

@author: Yingpeng_Du
"""
import pandas as pd 
import os
import numpy as np
path = 'D:\\乱七八糟的数据集\\assessmentdata\\assessmentdata\\data'
A= os.listdir(path)
res = []
for user in A:
    f = open(path+'\\'+user)
    line = f.readline()
    if line == 'No testing\n' or line == ' \n':
        f.close()
        continue
    while line:
        if line == ' \n' or line == '' or line == '   \n':
            line = f.readline()
            continue
        items = line.strip().split(' ')
        if items[1][0]=='M' and items[2] =='ON':
            res.append([user,items[1]])
        if items[1][0]=='I' and items[2] =='PRESENT':
            res.append([user,items[1]])
        if items[1][0]=='D' and items[2] =='OPEN':
            res.append([user,items[1]])
#        if items[1][0]=='T':
#            res.append([user,items[1]+str(int(float(items[2])/5))])
#        if items[1][0]=='A':
#            res.append([user,items[1]+str(int(float(items[2])/0.05))])
#        if items[1][0]=='P' and items[3] =='START' :
#            res.append([user,items[2]])                        
        line = f.readline()
    f.close()
n = 0
final = []
feature = [] 
for i,item in enumerate(res):
    if item[1][0] !='I' and item[1][0] !='D':
        feature.append('T'+item[1])
    else:
        final.append(['1']+item+feature[-5:])  
        feature.append('T'+item[1])
ratings = pd.DataFrame(final)
#时序
n=3
time = []
time_feature = []
values = ratings.values
for i in range(len(values)):
    try:
        values[i+1][1]
        if values[i][1] == values[i+1][1]:
            if len(time)==0:
                time_feature.append(['1' for j in range(n)])
            elif len(time)<n:
                time_feature.append(time + [time[-1] for j in range(n-len(time))])
            else:
                time_feature.append(time[-n:])
            time.append(values[i][2])
        if values[i][1] != values[i+1][1]:
            if len(time)==0:
                time_feature.append(['1' for j in range(n)])
            elif len(time)<n:
                time_feature.append(time + [time[-1] for j in range(n-len(time))])
            else:
                time_feature.append(time[-n:])
    except:
        if len(time)==0:
            time_feature.append(['1' for j in range(n)])
        elif len(time)<n:
            time_feature.append(time + [time[-1] for j in range(n-len(time))])
        else:
            time_feature.append(time[-n:])
time_feature = np.array(time_feature)    
for i in range(n,0,-1):
    ratings['time'+str(i)]  = list(map(lambda x: 'time'+x,time_feature[:,i-1]))
   

ratings[1:].to_csv('../data'+'/positive/jiaju/jiaju.libfm',index=False,header=None,sep=' ')     











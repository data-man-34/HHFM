# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:29:22 2018

@author: Yingpeng_Du
"""
import pandas as pd 
from AFM import AFM_main
from FM import FM_main
from OurModel7 import M7_main #HHFM
from DFM import DFM_main
from BPR import BPR_main
from CARS2 import CARS2_main
# =============================================================================
# Attention OurModel7 Line From 14~19 is the pooling operation. max sum mean
# =============================================================================
#data = 'frappe'
#itera = 1
#feature = ['raings','user','item','daytime','isweekend','homework']
#
##factors = [16,32,64,128]#
##for f in factors:
##  
#for fea in ['cost','weather','country','city','cnt']:
#    feature.append(fea)
#    ratings = pd.read_csv('file:///D:/乱七八糟的数据集/frappe/frappe/frappe.csv',sep='\t')
#    ratings = ratings.sort_values(['user'])
#    ratings['raings'] = 1
#    ratings['user']=ratings['user'].apply(lambda x: 'u'+str(x))
#    ratings['item']=ratings['item'].apply(lambda x: 'i'+str(x))
#    ratings['city']=ratings['city'].apply(lambda x: 'c'+str(x))
#    ratings['cnt'] =ratings['cnt'].apply(lambda x: 'cnt'+str(x))
#    ratings = ratings[feature]
#    ratings.to_csv('../data'+'/positive/frappe/frappe.libfm',index=False,header=None,sep=' ')   
#    f = 128    
#    M7_main(data,f,5)








 

datasets = ['jiaju' ,'resturant' ,'frappe']#,'tmall' ,'frappe' ,'jiaju' ,'resturant' ,          , 'resturant','frappe'    
factors = [128]# ,

for f in factors:
    for data in datasets:
        if data == 'jiaju':
            M7_main(data,f,1)
            AFM_main(data,f,1) 
            FM_main(data,f,1) 
            DFM_main(data,f,1)
            CARS2_main(data,f,1)
        else:
            M7_main(data,f,5)
            AFM_main(data,f,5)
            FM_main(data,f,5) 
            DFM_main(data,f,5)
            CARS2_main(data,f,5)                
            
#                
                
                
                
                
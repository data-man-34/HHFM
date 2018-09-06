# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:48:44 2018

@author: Yingpeng_Du
"""

import pandas as pd 
ratings = pd.read_csv('../data/rawdata/frappe/frappe/frappe.csv',sep='\t')
ratings = ratings.sort_values(['user'])
ratings['raings'] = 1
ratings['user']=ratings['user'].apply(lambda x: 'u'+str(x))
ratings['item']=ratings['item'].apply(lambda x: 'i'+str(x))
ratings['city']=ratings['city'].apply(lambda x: 'c'+str(x))
ratings['cnt'] =ratings['cnt'].apply(lambda x: 'cnt'+str(x))
ratings = ratings[['raings','user','item','daytime','isweekend','homework']]
#,'cost','weather','country','city','cnt'
ratings.to_csv('../data'+'/positive/frappe/frappe.libfm',index=False,header=None,sep=' ')     

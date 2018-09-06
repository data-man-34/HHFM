# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:11:11 2018

@author: Yingpeng_Du
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:40:12 2017

@author: QYH
"""
import numpy as np
string = '''
Iter=0[00:00:00.000] <loss, hr, ndcg, prec>:	 0.0000	 0.0047	 0.0009	 0.0002	 [00:00:03.888]
Iter=10[00:00:13.363] <loss, hr, ndcg, prec>:	 0.0000	 0.1867	 0.0482	 0.0186	 [00:00:36.793]
Iter=20[00:00:13.367] <loss, hr, ndcg, prec>:	 0.0000	 0.2284	 0.0591	 0.0227	 [00:00:44.668]
Iter=30[00:00:13.355] <loss, hr, ndcg, prec>:	 0.0000	 0.2457	 0.0652	 0.0263	 [00:00:48.306]
Iter=40[00:00:13.359] <loss, hr, ndcg, prec>:	 0.0000	 0.2552	 0.0686	 0.0282	 [00:00:49.775]
Iter=50[00:00:13.360] <loss, hr, ndcg, prec>:	 0.0000	 0.2630	 0.0718	 0.0302	 [00:00:50.728]
Iter=60[00:00:13.361] <loss, hr, ndcg, prec>:	 0.0000	 0.2682	 0.0732	 0.0308	 [00:00:51.367]
Iter=70[00:00:13.367] <loss, hr, ndcg, prec>:	 0.0000	 0.2688	 0.0732	 0.0306	 [00:00:51.742]
Iter=80[00:00:13.354] <loss, hr, ndcg, prec>:	 0.0000	 0.2699	 0.0734	 0.0304	 [00:00:51.579]
Iter=90[00:00:13.352] <loss, hr, ndcg, prec>:	 0.0000	 0.2748	 0.0753	 0.0317	 [00:00:52.300]
Iter=100[00:00:13.349] <loss, hr, ndcg, prec>:	 0.0000	 0.2757	 0.0753	 0.0315	 [00:00:52.429]
'''
string = string.split('\n')[1:-1]
save = []
for string2 in string:
    for i in range(5):
        string2 = string2.replace(' ','')    
    sting3 = string2.split('\t')
    save.append([sting3[2],sting3[3],sting3[4]])
save = np.array(save)   
    
    
    
    
#string = string.split('\n')[1:-1]
#save = []
#for string2 in string:
#    for i in range(5):
#        string2 = string2.replace('  ',' ')    
#    sting3 = string2.split(' ')        
#    save.append([float(sting3[i]) for i in range(1,10)])
#save = np.array(save)       
#
#    
    
    
    
    
    
    
    
    
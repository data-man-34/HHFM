# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:08:30 2018

@author: Yingpeng_Du
"""
import numpy as np
A='''Dataset=jiaju M7 Init: 	 train=AUC:0.4626;test=AUC:0.4259,HR:0.0217,NDCG:0.0217,PRE:0.0217;[0.7 s]
M7 Epoch 10 [0.7 s]	train=AUC:0.9674;test=AUC:0.9147,HR:0.5958,NDCG:0.5958,PRE:0.5958;[0.7 s]
M7 Epoch 20 [0.7 s]	train=AUC:0.9672;test=AUC:0.9147,HR:0.5692,NDCG:0.5692,PRE:0.5692;[0.9 s]
M7 Epoch 30 [0.8 s]	train=AUC:0.9735;test=AUC:0.9168,HR:0.6000,NDCG:0.6000,PRE:0.6000;[0.8 s]
M7 Epoch 40 [0.7 s]	train=AUC:0.9756;test=AUC:0.9187,HR:0.5925,NDCG:0.5925,PRE:0.5925;[0.9 s]
M7 Epoch 50 [0.7 s]	train=AUC:0.9729;test=AUC:0.9158,HR:0.5975,NDCG:0.5975,PRE:0.5975;[1.0 s]
Dataset=resturant M7 Init: 	 train=AUC:0.4923;test=AUC:0.4811,HR:0.0034,NDCG:0.0022,PRE:0.0018;[0.8 s]
M7 Epoch 10 [0.5 s]	train=AUC:0.9942;test=AUC:0.9605,HR:0.4510,NDCG:0.3085,PRE:0.2616;[0.7 s]
M7 Epoch 20 [0.6 s]	train=AUC:0.9966;test=AUC:0.9654,HR:0.4600,NDCG:0.3113,PRE:0.2627;[0.8 s]
M7 Epoch 30 [0.5 s]	train=AUC:0.9980;test=AUC:0.9649,HR:0.4789,NDCG:0.3221,PRE:0.2711;[0.8 s]
M7 Epoch 40 [0.5 s]	train=AUC:0.9989;test=AUC:0.9649,HR:0.4956,NDCG:0.3380,PRE:0.2864;[1.0 s]
M7 Epoch 50 [0.6 s]	train=AUC:0.9994;test=AUC:0.9636,HR:0.4900,NDCG:0.3310,PRE:0.2795;[0.9 s]
Dataset=frappe M7 Init: 	 train=AUC:0.4934;test=AUC:0.5048,HR:0.0079,NDCG:0.0075,PRE:0.0074;[1.0 s]
M7 Epoch 10 [3.9 s]	train=AUC:0.9986;test=AUC:0.9792,HR:0.6438,NDCG:0.5712,PRE:0.5474;[0.9 s]
M7 Epoch 20 [3.9 s]	train=AUC:0.9992;test=AUC:0.9812,HR:0.6739,NDCG:0.5979,PRE:0.5729;[1.0 s]
M7 Epoch 30 [4.0 s]	train=AUC:0.9994;test=AUC:0.9825,HR:0.6924,NDCG:0.6084,PRE:0.5808;[1.3 s]
M7 Epoch 40 [3.7 s]	train=AUC:0.9994;test=AUC:0.9810,HR:0.6894,NDCG:0.6144,PRE:0.5898;[1.2 s]'''
res = []
for line in A.split('\n'):
    sp = line.split(',')
    res.append([float(sp[0][-6:]), float(sp[1][-6:]),float(sp[2][-6:]),float(sp[3][4:10])])
res = np.array(res) 

#res = []
#for line in A.split('\n'):
#    sp = line.split('Epoch')
#    res.append([float(sp[1][1:4])])
#res = np.array(res) 









# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:08:14 2018

@author: Yingpeng_Du
"""

import tensorflow as tf
import numpy as np
sess = tf.Session()
A = tf.Variable(np.random.random([10,5,12]))
B = tf.Variable(np.random.random([20,10]))
C = tf.Variable(np.random.random([20,12]))
init = tf.global_variables_initializer()
sess.run(init)
W = tf.reshape(tf.matmul(B,tf.reshape(A,[10,-1])),[-1,5,12])
 tf.
d = sess.run(W)
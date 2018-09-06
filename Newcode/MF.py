
import math
import os, sys
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from time import time
import argparse
import Newcode.NewLoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import toolz
import copy
from tqdm import tqdm 
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--path', nargs='?', default='../data/positive/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='fra',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=110,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=256,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', type=float, default=0.7, 
                    help='Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    return parser.parse_args()

class MF(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, n_user , n_item ,  hidden_factor, learning_rate, lamda_bilinear, keep,
                 optimizer_type, batch_norm, verbose, random_seed=2016):
        # bind params to class
        self.n_user = n_user
        self.n_item = n_item
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name="train_features_fm")  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_fm")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep_fm")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase_fm")

            # Variables.
            self.weights = self._initialize_weights()
            #Model
            self.users = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,0], name='users') #None * k
            self.items = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,1], name='items') #None * k
            #self.features =  tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,1], name='features') #None * k
            self.FM_OUT = tf.multiply(self.users,self.items)  #None * K

            
            self.FM_OUT = tf.nn.dropout(self.FM_OUT, self.dropout_keep) # dropout at the FM layer

            # _________out _________
            self.Bilinear = tf.reduce_sum(self.FM_OUT, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features[:,:2]) , 1)  # None *2 * 1
            self.out = self.Bilinear #+ self.Feature_bias  # None * 1

            # Compute the square loss.
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.sess = self._init_session()
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)


    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
            name='feature_embeddings')  # features_M * K
        all_weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    def topk(self,A):
        user = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,0]) #None * k
        item = tf.nn.embedding_lookup(self.weights['feature_embeddings'], [i for i in range(self.n_user,self.n_user + self.n_item)]) # n * k
        score = tf.matmul(user,tf.transpose(item))  # none * n
        _, prediction = self.sess.run(tf.nn.top_k(score, 100), feed_dict = {self.train_features:A})         
        return prediction
        

class Train(object):
    def __init__(self,args):
        self.args = args
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.verbose = args.verbose
        self.keep = args.keep


    # Data loading
        self.data = DATA.LoadData(self.args.path, self.args.dataset)#获取数据
        self.n_user = self.data.n_user
        self.n_item = self.data.n_item
        if args.verbose > 0:
            print("FM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
                  %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm))

    # Training\\\

        self.model = MF(self.n_user+self.n_item, self.n_user, self.n_item , args.hidden_factor,args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm, args.verbose)

    def train(self):  # fit a dataset
        # Check Init performance
        #初始结果
        t2 = time()
        init_train_AUC = self.evaluate_AUC(self.data.Train_data)
        init_test_AUC = self.evaluate_AUC(self.data.Test_data)  
        init_test_TopK = self.evaluate_TopK(self.data.Test_data)
        
        if args.verbose > 0:
            print("Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]," %(init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
        for epoch in tqdm(range(1,self.epoch)):
            t1 = time()
            #sample负样本
            PosWithLable = self.data.Train_data.values
            PosWithLable_copy = copy.deepcopy(self.data.Train_data.values)           
            PosWithLable_copy = np.tile(np.expand_dims(PosWithLable_copy,axis =1),[1,2,1]) #none,10,3
            PosWithLable_copy = PosWithLable_copy.reshape([-1,PosWithLable_copy.shape[2]])#10*none,3
            NegSample = self.sample_negative(PosWithLable[:,1:],2)#几倍举例
            PosWithLable_copy[:,2] = NegSample.reshape([-1])
            PosWithLable_copy[:,0] = -1  # negative lable
            dat = np.append(PosWithLable,PosWithLable_copy,axis=0)
            np.random.shuffle(dat)
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(dat))] ):                
                X= np.array(dat[list(user_chunk)][:,1:],dtype = np.int)                
                Y = np.expand_dims(dat[list(user_chunk)][:,0],axis=1)
                batch_xs = {'X':X, 'Y':Y}               
                # Fit training
                self.model.partial_fit(batch_xs)
            t2 = time()

            # evaluate training and validation datasets

            if self.verbose > 0 and epoch%self.verbose == 0:
                init_train_AUC = self.evaluate_AUC(self.data.Train_data)
                init_test_AUC = self.evaluate_AUC(self.data.Test_data)        
                init_test_TopK = self.evaluate_TopK(self.data.Test_data)
       
                print("Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]"
                      %(epoch, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))

    def sample_negative(self, data,num=10):
        samples = np.random.randint( self.n_user,self.n_user + self.n_item,size = (len(data),num))
        for user, negatives, i in zip(data,
                                      samples,
                                      range(len(samples))):

            for j, neg in enumerate(negatives):
                key = tuple(user[[i for i in range(0,len(user)) if i!=1]])
                while neg in self.data.positive_feedback[key]:
                    samples[i, j] = neg = np.random.randint(self.n_user,self.n_user + self.n_item)
        return samples
    
    def evaluate_AUC(self, data1):  # evaluate the results for an input set
        dat = data1.values
        dat = dat[dat[:,0]>0] #去除标签为负的样本~
        
        data = {'X':np.array(dat[:,1:],dtype=np.int), 'Y':dat[:,0]}

        num_example = len(data['Y']) #需要验证样本的数量~
        

        score = []
        for user_chunk in toolz.partition_all(10000,[i for i in range(num_example)] ):
            pos = data['X'][list(user_chunk)]
            NegativeSample = self.sample_negative(pos)#采样负样本None,10
            neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,10,1]) #none,10,3
            neg = neg.reshape([-1,pos.shape[1]])#10*none,3
            neg[:,1] = NegativeSample.reshape([-1])#赋值给负样本，形成整个负样本集，注意到，这个地方10倍于正样本集        
            #计算negative评分
            feed_dict_neg = {self.model.train_features: neg, \
                             self.model.train_labels: [[1] for i in range(len(neg))],self.model.dropout_keep: 1.0, self.model.train_phase: False}
            self.neg_score = self.model.sess.run(self.model.out, feed_dict=feed_dict_neg)
            #计算positive评分
            feed_dict_pos = {self.model.train_features: pos, \
                             self.model.train_labels: [[1] for i in range(len(user_chunk))],self.model.dropout_keep:1.0, self.model.train_phase: False}
            batch_out_pos = self.model.sess.run( self.model.out, feed_dict=feed_dict_pos)
            self.pos_score = np.reshape(np.tile(np.expand_dims(batch_out_pos,axis = 1),[1,10,1]),[-1,1])
            
            score.extend(np.reshape(self.pos_score>self.neg_score,[-1]).tolist())
            
        return np.mean(score)
    def evaluate_TopK(self,data1):
        size = 1000 #验证其中的1000个
        result_MAP=[]
        result_NDCG=[]
        result_PRE = []
        dat = data1.values   
        
        for _ in range(int(size/100)):
            feed_dict = np.array(dat[:,1:][np.random.randint(0,len(dat),100)],dtype=np.int)
            self.score = self.model.topk(feed_dict)
            prediction = self.score + self.n_user
            for i,item in enumerate(feed_dict[:,1]):
                if item in prediction[i]:
                    result_MAP.append(1)
                    index1 = prediction[i].tolist().index(item)
                    result_NDCG.append(np.log(2)/np.log(index1+2))
                    result_PRE.append(1/(index1+1))
                else:
                    result_MAP.append(0)
                    result_NDCG.append(0)
                    result_PRE.append(0)        
        return  [np.average(result_MAP),np.average(result_NDCG),np.average(result_PRE)]
        


    

if __name__ == '__main__':
    args = parse_args()

    # if args.mla:
    #     args.lr = 0.1
    #     args.keep = '[1.0,1.0]'
    #     args.lamda_attention = 10.0
    # else:
    #     args.lr = 0.1
    #     args.keep = '[1.0,0.5]'
    #     args.lamda_attention = 100.0

    session = Train(args)
    session.train()
    session.evaluate(session.data.Test_data)

    
    
    
    
    
    
    
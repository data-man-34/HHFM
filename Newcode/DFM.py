
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
method = 'DFM'

def parse_args(dataname,factor,Topk):
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--path', nargs='?', default='../data/positive/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=dataname,
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=60,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', type=float, default=1, 
                    help='Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--TopK', type=int, default=Topk,
                    help='pass')
    parser.add_argument('--Result', type=int, default=0,
                    help='0:iteration 1:factors')
    return parser.parse_args()

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_user,
                 n_item,               
                 feature_size, 
                 field_size,
                 embedding_size, 
                 deep_layers,
                 deep_layers_activation,
                 learning_rate, 
                 verbose=True, 
                 l2_reg=0.0,
                 random_seed=2016,
                 use_fm=True,
                 use_deep=True,
                 loss_type="mse",
                 ):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        self.n_user = n_user
        self.n_item = n_item
        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.feat_index)  # None * F * K


            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
            self.y_first_order = tf.reduce_sum(self.y_first_order, 2)  # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)# None * F * K
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)#None * (F+K)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


            # init
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)



    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights



    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.feat_index: data['X'], self.label: data['Y']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    def topk(self,A,tp):
        pos = np.array(A,dtype=np.int)# none,3
        neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,self.n_item,1]) #none,N,3
        neg[:,:,1] = np.tile(np.expand_dims(np.array([i for i in range(self.n_user,self.n_user+self.n_item)]),axis =0),[neg.shape[0],1])        #none,N  
        neg = np.reshape(neg,[-1,self.field_size])  #none*N, 3 
        #计算negative评分
        self.neg_score =self.sess.run(self.out,feed_dict={self.feat_index:neg,  \
                                                          self.label: [[1] for i in range(len(neg))]}) # none*N
        #计算positive评分
        result = np.reshape(self.neg_score,[-1,self.n_item])
        sess = tf.Session()
        _, prediction = sess.run(tf.nn.top_k(result,tp))         
        return prediction  
        

class Train(object):
    def __init__(self,args):
        self.args = args
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.verbose = args.verbose
        self.keep = args.keep
        self.TopK = args.TopK


    # Data loading
        self.data = DATA.LoadData(self.args.path, self.args.dataset)#获取数据
        self.n_user = self.data.n_user
        self.n_item = self.data.n_item
        if args.verbose > 0:
            print("DFM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
                  %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm))



        # Training\\\
        self.model = DeepFM(self.n_user,self.n_item,self.data.features_M, self.data.Train_data.shape[1]-1,\
                            args.hidden_factor,[ 150,200,150],tf.nn.relu,\
                            args.lr,args.verbose,args.lamda)

    def train(self):  # fit a dataset
        # Check Init performance
        #初始结果
        t2 = time()
        if self.args.Result == 0:            
            init_train_AUC = self.evaluate_AUC(self.data.Train_data)
            init_test_AUC = self.evaluate_AUC(self.data.Test_data)  
            init_test_TopK = self.evaluate_TopK(self.data.Test_data)
            
            if self.args.verbose > 0:
                print("Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]," %(init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                with open("../result.txt","a") as f:
                    f.write("Dataset=%s %s Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n" %(self.args.dataset,method,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                    
        self.loss_epoch = []
        for epoch in tqdm(range(1,self.epoch)):
            loss = 0
            t1 = time()
            #sample负样本
            NG = 2
            PosWithLable = self.data.Train_data.values
            PosWithLable_copy = copy.deepcopy(self.data.Train_data.values)           
            PosWithLable_copy = np.tile(np.expand_dims(PosWithLable_copy,axis =1),[1,NG,1]) #none,10,3
            PosWithLable_copy = PosWithLable_copy.reshape([-1,PosWithLable_copy.shape[2]])#10*none,3
            
            NegSample = self.sample_negative(PosWithLable[:,1:],NG)#几倍举例
            PosWithLable_copy[:,2] = NegSample.reshape([-1])
            PosWithLable_copy[:,0] = -1 # negative lable
            dat = np.append(PosWithLable,PosWithLable_copy,axis=0)
            np.random.shuffle(dat)
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(dat))] ):                
                X= np.array(dat[list(user_chunk)][:,1:],dtype = np.int)                
                Y = np.expand_dims(dat[list(user_chunk)][:,0],axis=1)
                batch_xs = {'X':X, 'Y':Y}               
                # Fit training
                loss = loss + self.model.partial_fit(batch_xs)
            self.loss_epoch.append(loss)
            t2 = time()
            if self.args.Result == 1 and epoch>30:
                n=3
                condition = np.sum(( np.array(self.loss_epoch)[-1-n:-1]/np.array(self.loss_epoch)[-2-n:-2]-1)>-0.0075)
                if  condition==n:
                    init_train_AUC = self.evaluate_AUC(self.data.Train_data)
                    init_test_AUC = self.evaluate_AUC(self.data.Test_data)        
                    init_test_TopK = self.evaluate_TopK(self.data.Test_data)      
                    print("Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]"
                          %(epoch, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                    with open("../result.txt","a") as f:
                        f.write("%s%s Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n"
                          %(self.args.dataset,method,epoch, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                    break
            # evaluate training and validation datasets
            if self.args.Result == 0 and self.verbose > 0 and epoch%self.verbose == 0:
                init_train_AUC = self.evaluate_AUC(self.data.Train_data)
                init_test_AUC = self.evaluate_AUC(self.data.Test_data)        
                init_test_TopK = self.evaluate_TopK(self.data.Test_data)
       
                print("Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]"
                      %(epoch, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                with open("../result.txt","a") as f:
                    f.write("%s Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n"
                      %(method,epoch, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))

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
        for user_chunk in toolz.partition_all(600,[i for i in range(num_example)] ):
            pos = data['X'][list(user_chunk)]
            NegativeSample = self.sample_negative(pos,50)#采样负样本None,10
            neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,50,1]) #none,10,3
            neg = neg.reshape([-1,pos.shape[1]])#10*none,3
            neg[:,1] = NegativeSample.reshape([-1])#赋值给负样本，形成整个负样本集，注意到，这个地方10倍于正样本集        
            #计算negative评分
            feed_dict_neg = {self.model.feat_index: neg, \
                             self.model.label: [[1] for i in range(len(neg))]}
            self.neg_score = self.model.sess.run(self.model.out, feed_dict=feed_dict_neg)
            #计算positive评分
            feed_dict_pos = {self.model.feat_index: pos, \
                             self.model.label: [[1] for i in range(len(user_chunk))]}
            batch_out_pos = self.model.sess.run( self.model.out, feed_dict=feed_dict_pos)
            self.pos_score = np.reshape(np.tile(np.expand_dims(batch_out_pos,axis = 1),[1,50,1]),[-1,1])
            
            score.extend(np.reshape(self.pos_score>self.neg_score,[-1]).tolist())
            
        return np.mean(score)
    def evaluate_TopK(self,data1):
        size = np.min([3000,len(data1)]) #验证其中的1000个
        result_MAP=[]
        result_NDCG=[]
        result_PRE = []
        dat = data1.values   
        num = 60
        for _ in range(int(size/num)):
            feed_dict = np.array(dat[:,1:][np.random.randint(0,len(dat),num)],dtype=np.int)
            self.score = self.model.topk(feed_dict,20)
            prediction = self.score + self.n_user
            for i,line in enumerate(feed_dict):
                user,item = line[[0,1]]
                key = tuple(line[[i for i in range(0,len(line)) if i!=1]])

                n = 0 
                for it in prediction[i]:
                    if n> self.TopK -1:
                        result_MAP.append(0)
                        result_NDCG.append(0)
                        result_PRE.append(0)  
                        n=0
                        break
                    elif it == item:                        
                        result_MAP.append(1)
                        result_NDCG.append(np.log(2)/np.log(n+2))
                        result_PRE.append(1/(n+1))
                        n=0
                        break
                    elif item in self.data.positive_feedback[key]:
                        continue
                    else:
                        n = n +1   
      
        return  [np.average(result_MAP),np.average(result_NDCG),np.average(result_PRE)]
 
        


    
def DFM_main(dataname,factor,Topk):
    #dataname,factor,Topk=['jiaju',128,1]
    args = parse_args(dataname,factor,Topk)
    session = Train(args)
    session.train()


    
    
    
    
    
    
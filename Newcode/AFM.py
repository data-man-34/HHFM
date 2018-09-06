'''
Tensorflow implementation of Attentional Factorization Machines (AFM)

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Hao Ye (tonyfd26@gmail.com)

@references:
'''
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
method ='AFM'
sess = tf.Session()
def parse_args(dataname,factor,TopK):
    parser = argparse.ArgumentParser(description="Run DeepFM.")
    parser.add_argument('--path', nargs='?', default='../data/positive/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default= dataname,
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=60,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size.')
    parser.add_argument('--attention', type=int, default=1,
                        help='flag for attention. 1: use attention; 0: no attention')
    parser.add_argument('--hidden_factor', nargs='?', default='[%d,%d]'%(factor,factor),
                        help='Number of hidden factors.')
    parser.add_argument('--lamda_attention', type=float, default=100.0,
                        help='Regularizer for attention part.')
    parser.add_argument('--keep', nargs='?', default='[1,1]',
                        help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--decay', type=float, default=0.999,
                    help='Decay value for batch norm')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--TopK', type=int, default=TopK,
                    help='pass')
    parser.add_argument('--Result', type=int, default=0,
                    help='0:iteration 1:factors')    
    return parser.parse_args()

class AFM(BaseEstimator, TransformerMixin):
    def __init__(self, n_user,n_item,features_M, attention, hidden_factor, activation_function, 
                  learning_rate, lamda_attention, keep, optimizer_type, decay, valid_dimension ,random_seed=2016):
        # bind params to class
        self.n_user = n_user
        self.n_item = n_item
        self.learning_rate = learning_rate
        self.attention = attention
        self.hidden_factor = hidden_factor
        self.activation_function = activation_function
        self.features_M = features_M
        self.valid_dimension = valid_dimension
        self.lamda_attention = lamda_attention
        self.keep = keep
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.decay = decay
        self.u_f = valid_dimension - 1
        # performance of each epoch

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name="train_features_afm")  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_afm")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_afm")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase_afm")

            # Variables
            self.weights = self._initialize_weights()

            # Model.
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features) # None * M' * K
            
            element_wise_product_list = []
            count = 0
            for i in range(0, self.valid_dimension):
                for j in range(i+1, self.valid_dimension):
                    element_wise_product_list.append(tf.multiply(self.nonzero_embeddings[:,i,:], self.nonzero_embeddings[:,j,:]))
                    count += 1
            self.element_wise_product = tf.stack(element_wise_product_list) # (M'*(M'-1)) * None * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1,0,2], name="element_wise_product") # None * (M'*(M'-1)) * K
            # _________ MLP Layer / attention part _____________
            num_interactions = int(self.valid_dimension*(self.valid_dimension-1)/2)
            print(num_interactions)
            if self.attention:
                self.attention_mul = tf.reshape(tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.hidden_factor[1]]), \
                    self.weights['attention_W']), shape=[-1, num_interactions, self.hidden_factor[0]])
                # self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                #     self.weights['attention_b'])), 2, keep_dims=True)) # None * (M'*(M'-1)) * 1
                # self.attention_sum = tf.reduce_sum(self.attention_exp, 1, keep_dims=True) # None * 1 * 1
                # self.attention_out = tf.div(self.attention_exp, self.attention_sum, name="attention_out") # None * (M'*(M'-1)) * 1
                self.attention_relu = tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                    self.weights['attention_b'])), 2, keep_dims=True) # None * (M'*(M'-1)) * 1
                self.attention_out = tf.nn.softmax(self.attention_relu,axis =1)  #这个它地方程序写错了
                self.attention_out = tf.nn.dropout(self.attention_out, self.dropout_keep[0]) # dropout
            
            # _________ Attention-aware Pairwise Interaction Layer _____________
            if self.attention:
                self.AFM = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), 1, name="afm") # None * K
            else:
                self.AFM = tf.reduce_sum(self.element_wise_product, 1, name="afm") # None * K

            self.AFM = tf.nn.dropout(self.AFM, self.dropout_keep[1]) # dropout

            # _________ out _____________

            self.prediction = tf.matmul(self.AFM, self.weights['prediction']) # None * 1
            Bilinear = tf.reduce_sum(self.prediction, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias], name="out_afm")  # None * 1

            # Compute the loss.
            if self.attention and self.lamda_attention > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_attention)(self.weights['attention_W'])  # regulizer
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)


    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        all_weights = dict()
        # if freeze_fm, set all other params untrainable
# =============================================================================
#         flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; 
#-1: save to pretrain file; 2: initialize from pretrain and save to pretrain file
# =============================================================================
 
        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.features_M, self.hidden_factor[1]], 0.0, 0.01),
            name='feature_embeddings' )  # features_M * K
        all_weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        # attention
        if self.attention:
            glorot = np.sqrt(2.0 / (self.hidden_factor[0]+self.hidden_factor[1]))
            all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor[1], self.hidden_factor[0])), dtype=np.float32, name="attention_W")  # K * AK
            all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_factor[0])), dtype=np.float32, name="attention_b")  # 1 * AK
            all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.hidden_factor[0])), dtype=np.float32, name="attention_p") # AK

        # prediction layer
        all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor[1], 1), dtype=np.float32))  # hidden_factor * 1

        return all_weights



    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    def topk(self,A,tp):
        user = tf.expand_dims(tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,0]),axis=1) #None*1 * k
        feature = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,2:])#None*f * k
        user_feature = tf.concat([user,feature],axis=1) #None * (f+1) * k1
        #用户特征之间的交互作用
        element_wise_product_list = []
        count = 0
        for i in range(0, self.u_f):
            for j in range(i+1, self.u_f):
                element_wise_product_list.append(tf.multiply(user_feature[:,i,:], user_feature[:,j,:]))
                count += 1        
        uf = tf.transpose(tf.stack(element_wise_product_list),perm=[1,0,2]) #None * (u_f*(u_f-1)) * K1
        attention_mul_uf = tf.reshape(tf.matmul(tf.reshape(uf, shape=[-1, self.hidden_factor[1]]), \
            self.weights['attention_W']), shape=[-1, count, self.hidden_factor[0]])#None * (u_f*(u_f-1)) * K2
        aij_uf = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(attention_mul_uf + \
            self.weights['attention_b'])), 2, keep_dims=True)) # None * (u_f*(u_f-1)) * 1
        #项目（item）与（用户+特征）之间的交互作用
        item = tf.nn.embedding_lookup(self.weights['feature_embeddings'], [i for i in range(self.n_user,self.n_user + self.n_item)]) # n* k1
        uf_i = tf.multiply(tf.expand_dims(user_feature,axis=1),tf.expand_dims(tf.expand_dims(item,axis=0),axis=2)) #none * n *u-f *k1
        attention_mul_uf_i = tf.reshape(tf.matmul(tf.reshape(uf_i, shape=[-1, self.hidden_factor[1]]), \
            self.weights['attention_W']), shape=[-1, self.n_item, self.u_f , self.hidden_factor[0]])#none  * n * u-f *k2
        aij_ufi = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(attention_mul_uf_i + self.weights['attention_b'])), axis=3, keep_dims=True)) # none * n*u-f *1      
        #用户特征之间的交互作用——后续
        UFwise= tf.reduce_sum(tf.multiply(uf ,aij_uf),axis =1) #none* k1
        Iufwise =tf.reduce_sum(tf.multiply(uf_i,aij_ufi),axis = 2) #none*N*k1
        score1 = tf.expand_dims(UFwise,axis=1) + Iufwise #none*N*k1
        weight = tf.expand_dims(tf.reduce_sum(aij_uf,axis=1),axis=1)+tf.reduce_sum(aij_ufi,axis=2) # none*1 * 1 + none *n *1->none * n*1  
        score2 = tf.divide(score1,weight) #none*N*k1


        score3 = tf.reduce_sum(tf.multiply(score2, tf.transpose(self.weights['prediction'])),axis=2) # None * N
        bias = tf.transpose(tf.nn.embedding_lookup(self.weights['feature_bias'], [i for i in range(self.n_user,self.n_user + self.n_item)]) , perm=[1,0]) # 1 * n 

        
        score = score3 + bias   # None * N

        _, prediction = self.sess.run(tf.nn.top_k(score, tp), feed_dict = {self.train_features:A})         
        return  prediction
# =============================================================================
#         pos = np.array(A,dtype=np.int)# none,3
#         neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,self.n_item,1]) #none,N,3
#         neg[:,:,1] = np.tile(np.expand_dims(np.array([i for i in range(self.n_user,self.n_user+self.n_item)]),axis =0),[neg.shape[0],1])        #none,N  
#         neg = np.reshape(neg,[-1,self.valid_dimension])  #none*N, 3 
#         #计算negative评分
#         self.neg_score =self.sess.run(self.out,feed_dict={self.train_features:neg,  \
#                                                           self.train_labels: [[1] for i in range(len(neg))],self.dropout_keep: [1.0,1.0], self.train_phase: False}) # none*N
#         #计算positive评分
#         result = np.reshape(self.neg_score,[-1,self.n_item])
#         _, prediction = sess.run(tf.nn.top_k(result,tp))         
#         return prediction
# =============================================================================
        

        
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
        self.valid_dimension = self.data.Train_data.shape[1] - 1
        if args.verbose > 0:
            print("AFM: dataset=%s, factors=%s, #epoch=%d, batch=%d, lr=%.4f, lamda_attention=%.1e, keep=%s, optimizer=%s, batch_norm=%d"
                  %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda_attention, args.keep, args.optimizer, args.batch_norm))
        activation_function = tf.nn.relu
        if args.activation == 'sigmoid':
            activation_function = tf.sigmoid
        elif args.activation == 'tanh':
            activation_function == tf.tanh
        elif args.activation == 'identity':
            activation_function = tf.identity
    # Training\\    
        self.model = AFM(self.n_user,self.n_item,self.data.features_M, args.attention, eval(args.hidden_factor), 
            activation_function, args.lr, args.lamda_attention, eval(args.keep), args.optimizer, args.decay,self.valid_dimension)
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
            PosWithLable_copy[:,0] = -1  # negative lable
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
                condition = np.sum(( np.array(self.loss_epoch)[-1-n:-1]/np.array(self.loss_epoch)[-2-n:-2]-1)>-0.01)
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
            feed_dict_neg = {self.model.train_features: neg, \
                             self.model.train_labels: [[1] for i in range(len(neg))],self.model.dropout_keep: [1.0,1.0], self.model.train_phase: False}
            self.neg_score = self.model.sess.run(self.model.out, feed_dict=feed_dict_neg)
            #计算positive评分
            feed_dict_pos = {self.model.train_features: pos, \
                             self.model.train_labels: [[1] for i in range(len(user_chunk))],self.model.dropout_keep:[1.0,1.0], self.model.train_phase: False}
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
        num = 300
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
    
#        for _ in range(int(size/num)):
#            feed_dict = np.array(dat[:,1:][np.random.randint(0,len(dat),num)],dtype=np.int)
#            self.score = self.model.topk(feed_dict,self.TopK)
#            prediction = self.score + self.n_user
#            for i,item in enumerate(feed_dict[:,1]):
#                if item in prediction[i]:
#                    result_MAP.append(1)
#                    index1 = prediction[i].tolist().index(item)
#                    result_NDCG.append(np.log(2)/np.log(index1+2))
#                    result_PRE.append(1/(index1+1))
#                else:
#                    result_MAP.append(0)
#                    result_NDCG.append(0)
#                    result_PRE.append(0)  
        return  [np.average(result_MAP),np.average(result_NDCG),np.average(result_PRE)]
def AFM_main(dataname,factor,Topk):
#dataname,factor,Topk=['jiaju',128,1]
    args = parse_args(dataname,factor,Topk)
    session = Train(args)
    session.train()


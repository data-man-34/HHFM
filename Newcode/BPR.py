

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

method = 'BPR'
def parse_args(dataname,factor,Topk):
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--path', nargs='?', default='../data/positive/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=dataname,
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=1110,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', type=float, default=1, 
                    help='Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimize   r type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--TopK', type=int, default=Topk,
                    help='pass')
    parser.add_argument('--Result', type=int, default=1,
                    help='0:iteration 1:factors')
    return parser.parse_args()
class BPR(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, n_user , n_item ,  hidden_factor, learning_rate, lamda_bilinear,
                 optimizer_type):
        # bind params to class
        self.n_user = n_user
        self.n_item = n_item
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
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
            # Input data.
            self.Pos = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.Neg = tf.placeholder(tf.int32, shape=[None, None])  # None * 10
            
            # Variables.
            self.weights = self._initialize_weights()
            #Model
            self.user_select = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.Pos[:,0])    #None * k    
            self.pos_select = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.Pos[:,1])#None * k  
            self.neg_select = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.Neg)   #None*10 * k  
            self.PositiveFeadback = tf.reduce_sum(tf.multiply(self.user_select,self.pos_select),axis=1,keep_dims=True)#None*1 
            self.neg_item = tf.reduce_sum(tf.multiply(tf.expand_dims(self.user_select,axis=1),self.neg_select),axis=2) #None*10
            self.MaxNeg_item = tf.reduce_max(self.neg_item,axis=1,keep_dims=True) # None*1
            self.loss =   -tf.reduce_sum(tf.log(tf.sigmoid(self.PositiveFeadback- self.MaxNeg_item)))
      
            # Compute the square loss.
            if self.lamda_bilinear > 0:
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
            else:
                self.loss = self.loss
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
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.Pos: data['X'], self.Neg: data['Y']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    def topk(self,A,Topk):
        user = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.Pos[:,0]) #None * k
        item = tf.nn.embedding_lookup(self.weights['feature_embeddings'], [i for i in range(self.n_user,self.n_user + self.n_item)]) # n * k
        score = tf.matmul(user,tf.transpose(item))  # none * n
        _, prediction = self.sess.run(tf.nn.top_k(score, Topk), feed_dict = {self.Pos:A})         
        return prediction
    
    
class Train(object):
    def __init__(self,args):
        self.args = args
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.TopK = args.TopK

    # Data loading
        self.data = DATA.LoadData(self.args.path, self.args.dataset)#获取数据
        self.n_user = self.data.n_user
        self.n_item = self.data.n_item
        self.features_M = self.data.features_M
        self.valid_dimension = self.data.Train_data.shape[1] - 1
        print("OurModel: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
              %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm))






    # Training\\\建立模型
        self.model = BPR(self.features_M, self.n_user, self.n_item , args.hidden_factor,args.lr, args.lamda, args.optimizer)

    def train(self):  # fit a dataset
        # Check Init performance
        #初始结果            
        t2 = time()
        if self.args.Result == 0:            
            init_train_AUC = self.evaluate_AUC(self.data.Train_data)
            init_test_AUC = self.evaluate_AUC(self.data.Test_data)  
            init_test_TopK = self.evaluate_TopK(self.data.Test_data)
            
            print("Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]," %(init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
            with open("../result.txt","a") as f:
                f.write("Dataset=%s %s Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n," %(self.args.dataset,method,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                
        self.loss_epoch = []
 
        for epoch in tqdm(range(1,self.epoch)):
            loss = 0

            t1 = time()
            #sample负样本
            PosSample = self.data.Train_data.values[:,1:]
            np.random.shuffle(PosSample)
            NG = 10
            NegSample = self.sample_negative(PosSample,NG)#几倍举例
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(PosSample))] ):                
                X = np.array(PosSample[list(user_chunk)][:,:2],dtype = np.int) 
                Y = np.array(NegSample[list(user_chunk)],dtype = np.int)
                batch_xs = {'X':X,'Y':Y}      

                # Fit training
                loss = loss + self.model.partial_fit(batch_xs)
            self.loss_epoch.append(loss)
            t2 = time()
            if self.args.Result == 1 and epoch>30:
                n=3
                condition = np.sum(( np.array(self.loss_epoch)[-1-n:-1]/np.array(self.loss_epoch)[-2-n:-2]-1)>-0.0075)
                if  condition==n or epoch > 100:
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
            if self.args.Result == 0  and epoch% 10 == 0:
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
        data = data1.values[:,1:]
        num_example = len(data) #需要验证样本的数量~
        score = []
        for user_chunk in toolz.partition_all(600,[i for i in range(num_example)] ):
            pos = data[list(user_chunk)]
            NegativeSample = self.sample_negative(pos,50)#采样负样本None,10
            neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,50,1]) #none,10,3
            neg = neg.reshape([-1,pos.shape[1]])#10*none,3
            neg[:,1] = NegativeSample.reshape([-1])#赋值给负样本，形成整个负样本集，注意到，这个地方10倍于正样本集        
            #计算negative评分
            feed_dict_neg = {self.model.Pos: neg[:,:2]}
            #计算positive评分
            feed_dict_pos = {self.model.Pos: pos[:,:2]}

            batch_out_pos = self.model.sess.run( self.model.PositiveFeadback, feed_dict=feed_dict_pos)
            self.neg_score = self.model.sess.run(self.model.PositiveFeadback, feed_dict=feed_dict_neg)            
            self.pos_score = np.reshape(np.tile(np.expand_dims(batch_out_pos,axis = 1),[1,50,1]),[-1,1])
            
            score.extend(np.reshape(self.pos_score>self.neg_score,[-1]).tolist())

            
            return np.mean(score)
    def evaluate_TopK(self,data1):
        size = np.min([3000,len(data1)]) #验证其中的1000个
        result_MAP=[]
        result_NDCG=[]
        result_PRE = []
        dat = data1.values   
        num = 100
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
 
        
def BPR_main(dataname,factor,Topk):
#dataname,factor,Topk=['ml',128,5]
    args = parse_args(dataname,factor,Topk)
    session = Train(args)
    session.train()



    
    
    
    
    
        


    

    
    
    
    
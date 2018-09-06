

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

method = 'CARS2'
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
    parser.add_argument('--lamda', type=float, default=0.001,
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
    parser.add_argument('--Result', type=int, default=0,
                    help='0:iteration 1:factors')
    return parser.parse_args()
class CARS2(BaseEstimator, TransformerMixin):
    def __init__(self,features_M, n_user , n_item ,  hidden_factor, learning_rate, lamda_bilinear,
                 optimizer_type):
        # bind params to class

        self.n_user = n_user
        self.n_item = n_item
        self.learning_rate = learning_rate
        self.D = hidden_factor
        self.D_c = int(hidden_factor/2.5)
        self.D_p = int(hidden_factor/5)
        self.D_q = int(hidden_factor/2.5)
      
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.weights = self._initialize_weights()

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
            self.Pos = tf.placeholder(tf.int32, shape=[None, 2])  # None * 3
            self.Fea = tf.placeholder(tf.int32, shape=[None])  # None 
            self.Neg = tf.placeholder(tf.int32, shape=[None, None])  # None * 10

            
            # Variables.
            self.weights = self._initialize_weights()
            #Model user,item,context
            self.users = tf.nn.embedding_lookup(self.weights['UI'], self.Pos[:,0]) # None * D
            self.Positems = tf.nn.embedding_lookup(self.weights['UI'], self.Pos[:,1]) #None * D
            self.Negitems = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['UI'], self.Neg),1) #  None * D
            self.context = tf.nn.embedding_lookup(self.weights['Context'], self.Fea ) #None* D_c
            #pik
            self.pik_left = tf.reshape(tf.matmul(self.users,tf.reshape(self.weights['W'],[self.D,-1])),[-1,self.D_p,self.D_c]) #left  None* D_p*D_c
            self.pik = tf.reduce_sum(tf.multiply(self.pik_left,tf.expand_dims(self.context,axis=1)),axis=2) # None* D_p
            #qjk_pos
            self.qjk_left_pos = tf.reshape(tf.matmul(self.Positems,tf.reshape(self.weights['Z'],[self.D,-1])),[-1,self.D_q,self.D_c]) #left  None* D_q*D_c
            self.qjk_pos = tf.reduce_sum(tf.multiply(self.qjk_left_pos,tf.expand_dims(self.context,axis=1)),axis=2) # None* D_p
            #qjk_beg
            self.qjk_left_neg = tf.reshape(tf.matmul(self.Negitems,tf.reshape(self.weights['Z'],[self.D,-1])),[-1,self.D_q,self.D_c]) #left  None* D_q*D_c
            self.qjk_neg = tf.reduce_sum(tf.multiply(self.qjk_left_neg,tf.expand_dims(self.context,axis=1)),axis=2) # None* D_p
            
            


            #正样本            
            self.PositiveFeadback = (tf.reduce_sum(tf.multiply(self.users,self.Positems),axis=1,keep_dims=True) \
                                     +tf.reduce_sum(tf.multiply(self.pik,self.weights['A']),axis=1,keep_dims=True) \
                                     +tf.reduce_sum(tf.multiply(self.qjk_pos,self.weights['B']),axis=1,keep_dims=True))   #None*1
            #负样本
            self.NegativeFeadback = (tf.reduce_sum(tf.multiply(self.users,self.Negitems),axis=1,keep_dims=True) \
                                     +tf.reduce_sum(tf.multiply(self.pik,self.weights['A']),axis=1,keep_dims=True) \
                                     +tf.reduce_sum(tf.multiply(self.qjk_neg,self.weights['B']),axis=1,keep_dims=True))   #None*1


            self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(self.PositiveFeadback- self.NegativeFeadback)))
            
            # Compute the square loss.
            if self.lamda_bilinear > 0:
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['UI']) \
                + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['Context']) \
                + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['W']) \
                + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['Z']) \
                + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['A']) \
                + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['B']) 
                                                                
                
                
                # regulizer
            else:
                self.loss = self.loss
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
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)


    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['UI'] = tf.Variable(
            tf.random_normal([self.n_user+self.n_item,self.D], 0.0, 0.01))  
        all_weights['Context'] = tf.Variable(
            tf.random_normal([self.features_M,self.D_c], 0.0, 0.01))  
        all_weights['W'] = tf.Variable(
            tf.random_normal([self.D, self.D_p,self.D_c], 0.0, 0.01))  # 
        all_weights['Z'] = tf.Variable(
            tf.random_normal([self.D, self.D_q,self.D_c], 0.0, 0.01))  #        
        all_weights['A'] = tf.Variable(
            tf.random_uniform([self.D_p], 0.0, 0.0))  #  
        all_weights['B'] = tf.Variable(
            tf.random_uniform([self.D_q], 0.0, 0.0))  #  
        return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.Pos: data['X'], self.Fea:data['F1'] ,self.Neg: data['Y']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    def topk(self,feed_dict,tp):
        
# =============================================================================
        users = tf.nn.embedding_lookup(self.weights['UI'], feed_dict['X']) #None * D
        items = tf.nn.embedding_lookup(self.weights['UI'], [i for i in range(self.n_user,self.n_user + self.n_item)]) # n * D
        context = tf.nn.embedding_lookup(self.weights['Context'], feed_dict['F1']) #None * D_c
        #pik
        pik_left = tf.reshape(tf.matmul(users,tf.reshape(self.weights['W'],[self.D,-1])),[-1,self.D_p,self.D_c]) #left  None* D_p*D_c
        pik = tf.reduce_sum(tf.multiply(pik_left,tf.expand_dims(context,axis=1)),axis=2) # None* D_p
        #qjk_pos
        qjk_left_pos = tf.expand_dims(tf.reshape(tf.matmul(items,tf.reshape(self.weights['Z'],[self.D,-1])),[-1,self.D_q,self.D_c]),axis=0) #left 1* n* D_q*D_c
        qjk_pos = tf.reduce_sum(tf.multiply(qjk_left_pos,tf.expand_dims(tf.expand_dims(context,axis=1),axis=2)),axis=3) #none* n* D_q
        Feedback = tf.matmul(users,tf.transpose(items)) \
                    +tf.reduce_sum(tf.multiply(pik,self.weights['A']),axis=1,keep_dims=True) \
                    +tf.reduce_sum(tf.multiply(qjk_pos,self.weights['B']),axis=2)   #none*n +none*1  +  none*n

        _, prediction = self.sess.run(tf.nn.top_k(Feedback,tp ))         
# =============================================================================
#        pos = np.array(A,dtype=np.int)# none,3
#        neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,self.n_item,1]) #none,N,3
#        neg[:,:,1] = np.tile(np.expand_dims(np.array([i for i in range(self.n_user,self.n_user+self.n_item)]),axis =0),[neg.shape[0],1])        #none,N  
#        neg = np.reshape(neg,[-1,self.valid_dimension])  #none*N, 3 
#        #计算negative评分
#        self.neg_score =self.sess.run(self.PositiveFeadback,feed_dict={self.Pos:neg})
#        #计算positive评分
#        result = np.reshape(self.neg_score,[-1,self.n_item])
#        sess = tf.Session()
#        _, prediction = sess.run(tf.nn.top_k(result,tp))              
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
        set1 = set()
        for line in self.data.Total_data.values[:,3:]:
            set1.add(tuple(line))
        self.feature_inject = {f:i for i,f in enumerate(set1)}
        self.features_M = len(self.feature_inject.keys())
        print("OurModel: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
              %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm))

        self.model = CARS2(self.features_M, self.n_user, self.n_item , args.hidden_factor,args.lr, args.lamda, args.optimizer)

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
                f.write("Dataset=%s %s Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n" %(self.args.dataset,method,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
                
        self.loss_epoch = []
 
        for epoch in tqdm(range(1,self.epoch)):
            loss = 0

            t1 = time()
            #sample负样本
            PosSample = self.data.Train_data.values[:,1:]
            np.random.shuffle(PosSample)
            NG = 1
            NegSample = self.sample_negative(PosSample,NG)#几倍举例
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(PosSample))] ):                
                X = np.array(PosSample[list(user_chunk)][:,:2],dtype = np.int) 
                Y = np.array(NegSample[list(user_chunk)],dtype = np.int)     
                F1 = np.array(list(map(lambda x: self.feature_inject[tuple(x)],PosSample[list(user_chunk)][:,2:])),dtype = np.int) 
                batch_xs = {'X':X,'Y':Y,'F1':F1}  

                # Fit training
                loss = loss + self.model.partial_fit(batch_xs)
            self.loss_epoch.append(loss)
            t2 = time()
            if self.args.Result == 1 and epoch>10:
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
            F1 = np.array(list(map(lambda x: self.feature_inject[tuple(x)],neg[:,2:])),dtype = np.int) 
            F2 =  np.array(list(map(lambda x: self.feature_inject[tuple(x)],pos[:,2:])),dtype = np.int) 
            feed_dict_neg = {self.model.Pos: neg[:,:2],self.model.Fea:F1}
            #计算positive评分
            feed_dict_pos = {self.model.Pos: pos[:,:2],self.model.Fea:F2}
    
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
        num = 300
        for _ in range(int(size/num)):
            s = np.array(dat[:,1:][np.random.randint(0,len(dat),num)],dtype=np.int)
            X = s[:,0]        
            F1 = np.array(list(map(lambda x: self.feature_inject[tuple(x)],s[:,2:])),dtype = np.int) 
            feed_dict = {'X':X, 'F1':F1}
            self.score = self.model.topk(feed_dict,20)
            prediction = self.score + self.n_user
            for i,line in enumerate(s):
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
        
def CARS2_main(dataname,factor,Topk):
    #dataname,factor,Topk=['fra',16,5]
    args = parse_args(dataname,factor,Topk)
    session = Train(args)
    session.train()
                

    
    
    
    
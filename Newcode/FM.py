'''
Tensorflow implementation of Attentional Factorization Machines (AFM)

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Hao Ye (tonyfd26@gmail.com)

@references:
'''
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import Newcode.NewLoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import toolz
import copy
from tqdm import tqdm 
method = 'FM'


#################### Arguments ####################
def parse_args(dataname,factor,TopK):
    parser = argparse.ArgumentParser(description="Run DeepFM.")
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--mla', type=int, default=0,
                        help='Set the experiment mode to be Micro Level Analysis or not: 0-disable, 1-enable.')
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
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', type=float, default=1, 
                    help='Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--TopK', type=int, default=TopK,
                    help='pass')
    parser.add_argument('--Result', type=int, default=0,
                    help='0:iteration 1:factors')
        
    return parser.parse_args()

class FM(BaseEstimator, TransformerMixin):
    def __init__(self, valid_dimension,features_M,n_user,n_item , hidden_factor, learning_rate, lamda_bilinear, keep,
                 optimizer_type, batch_norm, verbose, random_seed=2016):
        # bind params to class
        self.valid_dimension = valid_dimension
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
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name="train_features_fm")  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_fm")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep_fm")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase_fm")

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # get the summed up embeddings of features.
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features, name='nonzero_embeddings') # None * 3 * K
            self.summed_features_emb = tf.reduce_sum(self.nonzero_embeddings, 1, keep_dims=True) # None * 1 * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * 1 * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(self.nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1, keep_dims=True)  # None * 1 * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb, name="fm")  # None * 1 * K
            # ml-tag has 3 interactions. divided by 3 to make sure that the sum of the weights is 1
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_OUT = tf.reduce_sum(self.FM, 1, name="fm_out") # None * K
            self.FM_OUT = tf.nn.dropout(self.FM_OUT, self.dropout_keep) # dropout at the FM layer

            # _________out _________
            Bilinear = tf.reduce_sum(self.FM_OUT, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias], name="out")  # None * 1

            # Compute the square loss.
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
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
        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
            name='feature_embeddings')  # features_M * K
        all_weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
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
    def topk(self,A,tp):
# =============================================================================
         user = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,0]) #None * k
         item = tf.nn.embedding_lookup(self.weights['feature_embeddings'], [i for i in range(self.n_user,self.n_user + self.n_item)]) # n * k
         feature = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:,2:]),axis=1) #None * k
         UserWithFeature = user + feature # None* k
         ItemWithFeature = tf.expand_dims(item,axis=0) + tf.expand_dims(feature,axis=1)   # none * n * k
         
         mul = tf.expand_dims(UserWithFeature,axis=1) * ItemWithFeature # none * n * k
         #mul = self.batch_norm_layer(mul, train_phase=False,scope_bn='')  # 测试
         
         score = tf.reduce_sum(mul,axis = 2)  # none * n
         bias = tf.transpose(tf.nn.embedding_lookup(self.weights['feature_bias'], [i for i in range(self.n_user,self.n_user + self.n_item)]) , perm=[1,0]) # 1 * n 
         _, prediction = self.sess.run(tf.nn.top_k(bias+score, tp), feed_dict = {self.train_features:A})         
# =============================================================================
#         pos = np.array(A,dtype=np.int)# none,3
#         neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,self.n_item,1]) #none,N,3
#         neg[:,:,1] = np.tile(np.expand_dims(np.array([i for i in range(self.n_user,self.n_user+self.n_item)]),axis =0),[neg.shape[0],1])        #none,N  
#         neg = np.reshape(neg,[-1,self.valid_dimension])  #none*N, 3 
#         #计算negative评分
#         self.neg_score =self.sess.run(self.out,feed_dict={self.train_features:neg,  \
#                                                           self.train_labels: [[1] for i in range(len(neg))],self.dropout_keep:[ 1.0], self.train_phase: False}) # none*N
#         #计算positive评分
#         result = np.reshape(self.neg_score,[-1,self.n_item])
#         sess = tf.Session()
#         _, prediction = sess.run(tf.nn.top_k(result,tp))         
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
        self.valid_dimension = self.data.Train_data.shape[1] - 1
        if args.verbose > 0:
            print("FM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
                  %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm))

    # Training\\\

        self.model = FM(self.valid_dimension,self.data.features_M,self.n_user,self.n_item,args.hidden_factor,args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm, args.verbose)

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
            PosWithLable_copy[:,0] = -0  # negative lable
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
                if  condition==n or epoch>100:
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
                             self.model.train_labels: [[1] for i in range(len(neg))],self.model.dropout_keep: 1.0, self.model.train_phase: False}
            self.neg_score = self.model.sess.run(self.model.out, feed_dict=feed_dict_neg)
            #计算positive评分
            feed_dict_pos = {self.model.train_features: pos, \
                             self.model.train_labels: [[1] for i in range(len(user_chunk))],self.model.dropout_keep:1.0, self.model.train_phase: False}
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
 
        
        
        

def FM_main(dataname,factor,Topk):
    #dataname,factor,Topk=['jiaju',128,1]
    args = parse_args(dataname,factor,Topk)
    session = Train(args)
    session.train()
    

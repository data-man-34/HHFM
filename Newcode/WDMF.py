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
import pandas as pd
method = 'WDMF'




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
    parser.add_argument('--epoch', type=int, default=110,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4096,
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
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--TopK', type=int, default=TopK,
                    help='pass')
    
    return parser.parse_args()

class WD(BaseEstimator, TransformerMixin):
    def __init__(self,feature_num,n_user,n_item):
        self.feature_num = feature_num
        self.n_user = n_user
        self.n_item = n_item
        self.keys = ["Feature"+str(i) for i in range(self.feature_num)]
        self.feature = [tf.contrib.layers.sparse_column_with_hash_bucket( key, hash_bucket_size=10e4)	for key in self.keys]
        self.interaction = []

        for i in range(len(self.feature)):
            for j in range(i+1,len(self.feature)):
                self.interaction.append([self.feature[i],self.feature[j]])
        
        
        self.wide_columns =  self.feature+ [tf.contrib.layers.crossed_column(element,hash_bucket_size=int(1e4)) for element in self.interaction]
        self.deep_columns = [tf.contrib.layers.embedding_column(self.feature[i], dimension=128)   for i,key in enumerate(self.keys)] 


        self.model = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir="",
        linear_feature_columns=self.wide_columns,
        dnn_feature_columns=self.deep_columns,
        dnn_hidden_units=[1024, 512 , 256])
        
        
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def input_fn(self,X,Y=0):
      """Input builder function."""
      # Creates a dictionary mapping from each continuous feature column name (k) to
      # the values of that column stored in a constant Tensor.
      #continuous_cols = {k: tf.constant(df[k].values) for k in self.model.keys}
      # Creates a dictionary mapping from each categorical feature column name (k)
      # to the values of that column stored in a tf.SparseTensor.
      df = pd.DataFrame(X,columns = self.keys,dtype= np.str)
      if type(Y)==int:
          df['label'] = np.zeros(len(X))
      else: 
          df['label'] = Y
      categorical_cols = {
          k: tf.SparseTensor(
              indices=[[i, 0] for i in range(df[k].size)],
              values=df[k].values,
              dense_shape=[df[k].size, 1]) for k in self.keys}
      # Merges the two dictionaries into one.
      feature_cols = dict()
      feature_cols.update(categorical_cols)
      # Converts the label column into a constant Tensor.
      label = tf.constant(df['label'].values)
      # Returns the feature columns and the label.
      return feature_cols, label        
        



    def predict(self,X):
        

        return  self.model.predict_proba(input_fn=lambda: self.input_fn(X))

    def partial_fit(self,X,Y):

        self.model.fit(input_fn=lambda: self.input_fn(X,Y), steps=500)
    def topk(self,feed_dict,tp):
        pos = np.array(feed_dict,dtype=np.int)# none,3
        neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,self.n_item,1]) #none,N,3
        neg[:,:,1] = np.tile(np.expand_dims(np.array([i for i in range(self.n_user,self.n_user+self.n_item)]),axis =0),[neg.shape[0],1])        #none,N  
        self.neg = np.reshape(neg,[-1,self.feature_num])  #none*N, 3 
        #计算negative评分
        self.neg_score = np.array(list(self.predict(self.neg)))[:,1] # none*N
        #计算positive评分
        result = np.reshape(self.neg_score,[-1,self.n_item])
        _, prediction = self.sess.run(tf.nn.top_k(result,tp))         
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
            print("FM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
                  %(args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer, args.batch_norm))

    # Training\\\

        self.model = WD(self.data.Train_data.shape[1]-1 ,self.n_user,self.n_item)

    def train(self):  # fit a dataset
        # Check Init performance
        #初始结果
        t2 = time()
#        init_train_AUC = self.evaluate_AUC(self.data.Train_data)
#        
#        print(init_train_AUC)
#        init_test_AUC = self.evaluate_AUC(self.data.Test_data)  
#        init_test_TopK = self.evaluate_TopK(self.data.Test_data)
#        
        if self.args.verbose > 0:
            print("Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]," %(0,0,0,0, 0,time()-t2))
            with open("../result.txt","a") as f:
                f.write("Dataset=%s %s Init: \t train=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n," %(self.args.dataset,method,0,0,0,0,0, time()-t2))
        for epoch in tqdm(range(1,11)):
            t1 = time()
            #sample负样本
            NG = 1
            PosWithLable = self.data.Train_data.values
            PosWithLable_copy = copy.deepcopy(self.data.Train_data.values)           
            PosWithLable_copy = np.tile(np.expand_dims(PosWithLable_copy,axis =1),[1,NG,1]) #none,10,3
            PosWithLable_copy = PosWithLable_copy.reshape([-1,PosWithLable_copy.shape[2]])#10*none,3
            
            NegSample = self.sample_negative(PosWithLable[:,1:],NG)#几倍举例
            PosWithLable_copy[:,2] = NegSample.reshape([-1])
            PosWithLable_copy[:,0] = 0  # negative lable
            dat = np.append(PosWithLable,PosWithLable_copy,axis=0)
            np.random.shuffle(dat)
            
            
            X= np.array(dat[:,1:],dtype = np.int)                
            Y =dat[:,0]

            # Fit training
            self.model.partial_fit(X,Y)
            t2 = time()
             
            # evaluate training and validation datasets

            init_train_AUC = self.evaluate_AUC(self.data.Train_data)
            init_test_AUC = self.evaluate_AUC(self.data.Test_data)        
            init_test_TopK = self.evaluate_TopK(self.data.Test_data,)
   
            print("Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]"
                  %(epoch*10, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))
            with open("../result.txt","a") as f:
                f.write("%s Epoch %d [%.1f s]\ttrain=AUC:%.4f;test=AUC:%.4f,HR:%.4f,NDCG:%.4f,PRE:%.4f;[%.1f s]\n"
                  %(method,epoch*10, t2-t1,init_train_AUC,init_test_AUC,init_test_TopK[0],init_test_TopK[1],init_test_TopK[2], time()-t2))

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
        for user_chunk in toolz.partition_all(3000,[i for i in range(num_example)] ):

            pos = data['X'][list(user_chunk)]
            NegativeSample = self.sample_negative(pos)#采样负样本None,10
            neg = np.tile(np.expand_dims(copy.deepcopy(pos),axis =1),[1,10,1]) #none,10,3
            neg = neg.reshape([-1,pos.shape[1]])#10*none,3
            neg[:,1] = NegativeSample.reshape([-1])#赋值给负样本，形成整个负样本集，注意到，这个地方10倍于正样本集        
            #计算negative评分
            neg_score = np.array(list(self.model.predict(neg)))[:,1]
            #计算positive评分
            batch_out_pos = np.array(list(self.model.predict(pos)))[:,1]
            pos_score= np.reshape(np.tile(np.expand_dims(batch_out_pos,axis = 1),[1,10,1]),[-1,1])
            score.append(np.mean(pos_score>neg_score))
        
        return np.mean(score)
    def evaluate_TopK(self,data1):
        size = 500 #验证其中的1000个
        result_MAP=[]
        result_NDCG=[]
        result_PRE = []
        dat = data1.values   
        
        for _ in range(int(size/25)):
            feed_dict = np.array(dat[:,1:][np.random.randint(0,len(data1),50)],dtype=np.int)
            self.score = self.model.topk(feed_dict,self.TopK)
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

        
        
        
        
        

def WDMF_main(dataname,factor,Topk):
#dataname,factor,Topk = ['jiaju',128,5]
    args = parse_args(dataname,factor,Topk)
    session = Train(args)
    session.train()




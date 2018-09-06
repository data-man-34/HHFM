
import numpy as np
import os
import pandas as pd
from collections import defaultdict
class LoadData(object):
    '''given the path of data, return the data format for AFM and FM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset,ratio = 0.9):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset +".libfm"
        self.Total_data = pd.read_csv(self.trainfile,sep=' ', header=None)
        self.Total_data.columns = ['label','user','item']+['feature'+str(i-2) for i in range(3,self.Total_data.shape[1])]
        #由于ITEM和USER不是连续的，我们需要对其做连续化处理
        self.n_user = len(self.Total_data['user'].value_counts())
        self.n_item = len(self.Total_data['item'].value_counts())

        injectset = set()
        injectdict =dict()
        num = 0
    
        for item in self.Total_data.values[:,1:].T.reshape([-1]):
            if item not in injectset:
                injectset.add(item)
                injectdict[item] = num
                num = num+1
        self.Total_data[self.Total_data.columns[1:]]= self.Total_data[self.Total_data.columns[1:]].applymap(lambda x : injectdict[x]) 
        self.features_M = len(injectset)

        #做AUC用的采样
        data = self.Total_data.values
        np.random.shuffle(data)
        test_size = int(len(data) * (1-ratio))
        self.Train_data = []
        self.Test_data = []
        self.positive_feedback = defaultdict(set)#用来存放
        self.train_set = defaultdict(set)
        set_key = set()
        i = 0
        #train
        for line in data:
            key = tuple(line[[i for i in range(1,len(line)) if i!=2]])
            indice = line[2]
            if key not in set_key and i<test_size:
                set_key.add(key)
                self.Test_data.append(line) 
                i = i + 1
            else:   
                self.positive_feedback[key].add(indice)
                self.Train_data.append(line)
                self.train_set[line[1]].add(line[2])
        self.Train_data = pd.DataFrame(np.array(self.Train_data))
        self.Test_data = pd.DataFrame(np.array(self.Test_data))
        self.Train_data.columns = self.Total_data.columns
        self.Test_data.columns = self.Total_data.columns





# 內建库
import sys
import heapq
# 第三方库
import sklearn.metrics as skm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




def emb_plot(data, legend, metrics,tl,name):
    '''
    label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    线型：-  --   -.  :    , 
    marker：.  ,   o   v    <    *    +    1
    '''
    plt.figure(figsize=(10,5))
    plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框

    line_store = ('-' , '--' , '-.' , ':' , '--')
    marker_store = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    
    x = data.index.levels[0]
    for i in range(len(legend)):
        y = data.xs(legend[i], level=1).loc[:, metrics]
        plt.plot(x, y, label=legend[i], ls=line_store[i], marker=marker_store[i])  
    plt.xlabel('alpha', fontsize=25)
    plt.xticks(x, map(str, x),fontsize = 20)
    #plt.xlim(0, 105)
    plt.yticks(fontsize = 20)
    
    plt.ylabel(metrics.upper(), fontsize=25)
    #plt.ylim(0, 1)

    plt.legend(loc='lower right', framealpha=0.5,fontsize = 20)
    # plt.legend(loc=0, numpoints=1)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=12,fontweight='bold')
    plt.title(tl,fontsize = 30)
    plt.savefig('../画图/%s.png' % name)  #建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()
    return
     
# NDCG
if __name__ == '__main__':
    for index in ['HR','NDCG']:
        
        name = 'WindowSize'
        #index = 'NDCG'
        df = pd.read_csv('file:///D:/鹏哥/代码/AFM/画图/s_embed.csv', index_col=(0,1))
        emb_plot(df,  ['Yelp','Netflix','MovieLens','Last.fm'], index,name,name+index)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
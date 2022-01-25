# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:08:23 2021

@author: BM109X32G-10GPU-02
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:46:29 2020

@author: de''
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:40:57 2020

@author: de''
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
import json
import numpy as np
import math
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def split_smiles(smiles, kekuleSmiles=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    except:
        pass
    splitted_smiles = []
    for j, k in enumerate(smiles):
        if len(smiles) == 1:
            return [smiles]
        if j == 0:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            else:
                splitted_smiles.append(k)
        elif j != 0 and j < len(smiles) - 1:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            elif k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)

        elif j == len(smiles) - 1:
            if k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
    return splitted_smiles

def get_maxlen(all_smiles, kekuleSmiles=True):
    maxlen = 0
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        maxlen = max(maxlen, len(spt))
    return maxlen
def get_dict(all_smiles, save_path, kekuleSmiles=True):
    words = [' ']
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        for w in spt:
            if w in words:
                continue
            else:
                words.append(w)
    with open(save_path, 'w') as js:
        json.dump(words, js)
    return words

def one_hot_coding(smi, words, kekuleSmiles=True, max_len=1000):
    coord_j = []
    coord_k = []
    spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
    if spt is None:
        return None
    for j,w in enumerate(spt):
        if j >= max_len:
            break
        try:
            k = words.index(w)
        except:
            continue
        coord_j.append(j)
        coord_k.append(k)
    data = np.repeat(1, len(coord_j))
    output = sparse.csr_matrix((data, (coord_j, coord_k)), shape=(max_len, len(words)))
    return output
def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
   # np.random.seed(111)  # fix the seed for shuffle.
    #np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = [np.arange(3)]
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
def edit_dataset(drug,non_drug,task):
  #  np.random.seed(111)  # fix the seed for shuffle.

#    np.random.shuffle(non_drug)
    non_drug=non_drug[0:len(drug)]
       

      #  np.random.shuffle(non_drug)
   # np.random.shuffle(drug)
    dataset_train_drug, dataset_test_drug = split_dataset(drug, 0.9)
   # dataset_train_drug,dataset_dev_drug =  split_dataset(dataset_train_drug, 0.9)
    dataset_train_no, dataset_test_no = split_dataset(non_drug, 0.9)
   # dataset_train_no,dataset_dev_no =  split_dataset(dataset_train_no, 0.9)
    dataset_train =  pd.concat([dataset_train_drug,dataset_train_no], axis=0)
    dataset_test=pd.concat([ dataset_test_drug,dataset_test_no], axis=0)
  #  dataset_dev = dataset_dev_drug+dataset_dev_no
    return dataset_train, dataset_test
if __name__ == "__main__":
    data_train= pd.read_csv('H:/qdf/qdf/dataset/transf/train.csv')
    data_test=pd.read_csv('H:/qdf/qdf/dataset/transf/test.csv')
    inchis = list(data_train['SMILES'])
    rts = list(data_train['PCE'])
    
    smiles, targets = [], []
    for i, inc in enumerate(tqdm(inchis)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)
            targets.append(rts[i])
            
    words = get_dict(smiles, save_path='E:\code\FingerID Reference\drug-likeness/dict.json')
    
    features = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=600)
        if xi is not None:
            features.append(xi.todense())
    features = np.asarray(features)
    targets = np.asarray(targets)
    X_train=features
    Y_train=targets
      

   # physical_devices = tf.config.experimental.list_physical_devices('CPU') 
   # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  #  tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    
  
    inchis = list(data_test['SMILES'])
    rts = list(data_test['PCE'])
    
    smiles, targets = [], []
    for i, inc in enumerate(tqdm(inchis)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)
            targets.append(rts[i])
            
   # words = get_dict(smiles, save_path='D:/工作文件/work.Data/CNN/dict.json')
    
    features = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=600)
        if xi is not None:
            features.append(xi.todense())
    features = np.asarray(features)
    targets = np.asarray(targets)
    X_test=features
    Y_test=targets
    n_features=10
    
    model = MLPRegressor()
    #model = MLPClassifier(rangdom_state=1,max_iter=300)
    #model = SVC()
   
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=0.05, verbose=0, mode='min')
    #mcp_save = ModelCheckpoint('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/single_model.h5', save_best_only=True, monitor='accuracy', mode='auto')
  #  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    from tensorflow.keras import backend as K
    X_train = K.cast_to_floatx(X_train).reshape((np.size(X_train,0),np.size(X_train,1)*np.size(X_train,2)))

    Y_train = K.cast_to_floatx(Y_train)
    
#    X_train,Y_train = make_blobs(n_samples=300, n_features=n_features, centers=6)
    model.fit(X_train, Y_train)

    
 #   model = load_model('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/single_model.h5')
    Y_predict = model.predict(K.cast_to_floatx(X_test).reshape((np.size(X_test,0),np.size(X_test,1)*np.size(X_test,2))))
     #Y_predict = model.predict(X_test)#训练数据
    x = list(Y_test)
    y = list(Y_predict)
    cc = np.array([x, y])
    cc_zscore_corr = np.corrcoef(cc)
    from scipy.stats import pearsonr
    print(pearsonr(x,y))
    
    r2 = r2_score(x,y)
    mae = mean_absolute_error(x,y)
    medae = median_absolute_error(x,y)
   
    from scipy.stats import pearsonr
    print(pearsonr(x,y))
    classes = ['A', 'B', 'C']
    A=[]
    B=[]
    C=0
    D=0
    for a in y:
        if a <3:
            a=1
            A.append(a)
            
        elif a<9 :
            a=2
            A.append(a)
        # elif a<9:
        #     a=3
        #     A.append(a)
        elif a>8.5:
            
            a=3
            A.append(a)
    for a in x:
        if a <3:
            a=1
            B.append(a)
        # elif a <6:
        #     a=2
        #     B.append(a)
        elif a<9:
            a=2
            B.append(a)
        elif a>8.5:
            
            a=3
            B.append(a)
    # 获取混淆矩阵
    random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
    y_true = random_numbers.copy()  # 样本实际标签
    random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
    y_pred = random_numbers  # 样本预测标签
    
    A=np.array(A)
    B=np.array(B)
    cm = confusion_matrix(A, B)
    plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
    print(cm_normalized)
  #  X= pd.concat([x,y], axis=1)
    #X.to_csv('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/molecularGNN_smiles-master/0825/single-CNN-seed444.csv')
    #Y_predict = [1 if i >0.4 else 0 for i in Y_predict]


import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
# read file
def generate_dataset (datapath,trainout,valout,testout):
    dataset = datapath
    data = pd.read_csv(dataset)
    path = '../dataset/new/'
   
    data["PCE"] = np.floor(data["PCE"])
    X =data['SMILES']
    y = data['PCE']    
    ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state=42)
    
    for train_index, test_index in ss.split(X, y):
        print("TRAIN_INDEX:", train_index, "TEST_INDEX:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print("X_train:",X_train)
        print("y_train:",y_train)
    ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state=21)
    da = pd.read_csv(dataset)
    
    test = da.iloc[test_index,:]
    test=test.reset_index(drop=True)
    test.to_csv(path+testout+'.csv')
    test.to_csv(path+testout+'.txt',index=False)
    train1 = da.iloc[train_index,:]
    da = da.iloc[train_index,:]
    da=da.reset_index(drop=True)
    # da=da.reset_index(drop=True)
    
    train1=train1.reset_index(drop=True)
    train1["PCE"] = np.floor(train1["PCE"])
    X =train1['SMILES']
    y = train1['PCE']
    
    
    for train_index, val_index in ss.split(X, y):
        print("TRAIN_INDEX:", train_index, "TEST_INDEX:", val_index)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        print("X_train:",X_train)
        print("y_train:",y_train)
        
    train=da.iloc[train_index,:]
    train=train.reset_index(drop=True)
    train.to_csv(path+trainout+'.csv')
    train.to_csv(path+trainout+'.txt',index=False)
    val=da.iloc[val_index,:]
    val=val.reset_index(drop=True)
    val.to_csv(path+valout+'.csv')
    val.to_csv(path+valout+'.txt')
    return print('datasets have generated!')
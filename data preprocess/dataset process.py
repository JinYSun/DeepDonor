import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
# 读取数据
data = pd.read_csv("E:/code/computionalmaterial/donorsm/us/newsm/opv.csv")
# 查看原始数据在数据中的比例
X =data['SMILES']
y = data['PCE']

data["PCE"] = np.floor(data["PCE"])
#data["PCE"].where(data["income_cat"]<5, 5.0, inplace=True)


ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state=42)#分成2组，测试比例为0.25

for train_index, test_index in ss.split(X, y):
    print("TRAIN_INDEX:", train_index, "TEST_INDEX:", test_index)#获得索引值
    X_train, X_test = X[train_index], X[test_index]#训练集对应的值
    y_train, y_test = y[train_index], y[test_index]#类别集对应的值
    
    print("X_train:",X_train)
    print("y_train:",y_train)
ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state=21)#分成2组，测试比例为0.25
da = pd.read_csv("E:/code\computionalmaterial/donorsm/us/newsm/opv.csv")

test = da.iloc[test_index,:]
test=test.reset_index(drop=True)
test.to_csv('H:/qdf\qdf\dataset/transf/test0.csv')
test.to_csv('H:/qdf\qdf\dataset/transf/test0.txt',index=False)
train1 = da.iloc[train_index,:]
da = da.iloc[train_index,:]
da=da.reset_index(drop=True)
# da=da.reset_index(drop=True)

train1=train1.reset_index(drop=True)
train1["PCE"] = np.floor(train1["PCE"])
X =train1['SMILES']
y = train1['PCE']


for train_index, val_index in ss.split(X, y):
    print("TRAIN_INDEX:", train_index, "TEST_INDEX:", val_index)#获得索引值
    X_train, X_val = X[train_index], X[val_index]#训练集对应的值
    y_train, y_val = y[train_index], y[val_index]#类别集对应的值
    
    print("X_train:",X_train)
    print("y_train:",y_train)
    
train=da.iloc[train_index,:]
train=train.reset_index(drop=True)
train.to_csv('H:/qdf\qdf\dataset/transf//train0.csv')
train.to_csv('H:/qdf\qdf\dataset/transf//train0.txt',index=False)
val=da.iloc[val_index,:]
val=val.reset_index(drop=True)
val.to_csv('H:/qdf\qdf\dataset/transf/val0.csv')
val.to_csv('H:/qdf\qdf\dataset/transf/val0.txt')
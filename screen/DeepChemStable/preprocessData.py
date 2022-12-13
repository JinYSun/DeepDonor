import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

def splitTrainDevTest(file, mark):
    """
    Origin file should be:
     SMILES Molname
    
    split:
    8:1:1
    """
    df = pd.read_table(file, header=None, index_col=1)
    df.columns = ["SMILES"]
    if mark == "pos": df["Type"] = 1
    elif mark == "neg": df["Type"] = 0
    else: raise ValueError
    
    idx = list(df.index)
    random.seed(321)
    random.shuffle(idx)
    idx_train = idx[:int(0.8*len(idx))]
    idx_dev = idx[int(0.8*len(idx)):int(0.9*len(idx))]
    idx_test = idx[int(0.9*len(idx)):]
    
    df_train = df.ix[idx_train]
    df_dev = df.ix[idx_dev]
    df_test = df.ix[idx_test]
    
    return df_train, df_dev, df_test

def createTrainDevTest(pos_file, neg_file):

    df_train_pos, df_dev_pos, df_test_pos = splitTrainDevTest(pos_file, "pos")
    df_train_neg, df_dev_neg, df_test_neg = splitTrainDevTest(neg_file, "neg")
    
    # merge
    df_train = pd.concat([df_train_pos, df_train_neg], axis=0)
    df_dev = pd.concat([df_dev_pos, df_dev_neg], axis=0)
    df_test = pd.concat([df_test_pos, df_test_neg], axis=0)
    
    return df_train, df_dev, df_test

df_train, df_dev, df_test = createTrainDevTest("data/Unstable3304.smi", "data/Stable6442.smi")
colnames = ["smiles", "label", "substance_id"]

df_train["substance_id"] = df_train.index
df_train.columns = colnames
df_train = df_train.iloc[:, [2, 0, 1]]

df_dev["substance_id"] = df_dev.index
df_dev.columns = colnames
df_dev = df_dev.iloc[:, [2, 0, 1]]

df_test["substance_id"] = df_test.index
df_test.columns = colnames
df_test = df_test.iloc[:, [2, 0, 1]]

del_list = ["C00735", "C03159", "C03161", "C05131", "C01592"]
df_train = df_train[~df_train.substance_id.isin(del_list)]
df_dev = df_dev[~df_dev.substance_id.isin(del_list)]
df_test = df_test[~df_test.substance_id.isin(del_list)]

df_train = df_train.sample(len(df_train))
df_dev = df_dev.sample(len(df_dev))
df_test = df_test.sample(len(df_test))

df_train.to_csv("data/train"+str(len(df_train))+".csv", index=None)
df_dev.to_csv("data/dev"+str(len(df_dev))+".csv", index=None)
df_test.to_csv("data/test"+str(len(df_test))+".csv", index=None)

df_all = pd.concat([df_train, df_dev, df_test], axis=0)
df_all.to_csv("data/Stable_Unstable_shuffled"+str(len(df_all))+".csv", index=None)

df_train["smiles"].to_csv("data/train_smi.csv", index=None)
df_dev["smiles"].to_csv("data/dev_smi.csv", index=None)
df_test["smiles"].to_csv("data/test_smi.csv", index=None)

train_smi = Chem.SmilesMolSupplier("data/train_smi.csv")
dev_smi = Chem.SmilesMolSupplier("data/dev_smi.csv")
test_smi = Chem.SmilesMolSupplier("data/test_smi.csv")

train_smi_fp = []
for i in range(len(train_smi)):
    train_smi_fp.append(FingerprintMols.FingerprintMol(train_smi[i]))

    dev_smi_fp = []
for i in range(len(dev_smi)):
    dev_smi_fp.append(FingerprintMols.FingerprintMol(dev_smi[i]))

test_smi_fp = []
for i in range(len(test_smi)):
    test_smi_fp.append(FingerprintMols.FingerprintMol(test_smi[i]))

#test vs train
test_train_simis = []
for i in range(len(test_smi_fp)):
    res = []
    for j in range(len(train_smi_fp)):
        res.append(DataStructs.FingerprintSimilarity(test_smi_fp[i],train_smi_fp[j]))   
    test_train_simis.append(np.mean(np.array(res)))
    
#test vs dev
test_dev_simis = []
for i in range(len(test_smi_fp)):
    res = []
    for j in range(len(dev_smi_fp)):
        res.append(DataStructs.FingerprintSimilarity(test_smi_fp[i],dev_smi_fp[j]))   
    test_dev_simis.append(np.mean(np.array(res)))
    
#train vs dev
train_dev_simis = []
for i in range(len(train_smi_fp)):
    res = []
    for j in range(len(dev_smi_fp)):
        res.append(DataStructs.FingerprintSimilarity(train_smi_fp[i],dev_smi_fp[j]))   
    train_dev_simis.append(np.mean(np.array(res)))

np.mean(test_train_simis)
np.mean(test_dev_simis)
np.mean(train_dev_simis)

#train
train_train_simis = []
for i in range(len(train_smi_fp)):
    res = []
    for j in range(len(train_smi_fp)):
        res.append(DataStructs.FingerprintSimilarity(train_smi_fp[i],train_smi_fp[j]))   
    train_train_simis.append(np.mean(np.array(res)))
    
#dev
dev_dev_simis = []
for i in range(len(dev_smi_fp)):
    res = []
    for j in range(len(dev_smi_fp)):
        res.append(DataStructs.FingerprintSimilarity(dev_smi_fp[i],dev_smi_fp[j]))   
    dev_dev_simis.append(np.mean(np.array(res)))
    
#test
test_test_simis = []
for i in range(len(test_smi_fp)):
    res = []
    for j in range(len(test_smi_fp)):
        res.append(DataStructs.FingerprintSimilarity(test_smi_fp[i],test_smi_fp[j]))   
    test_test_simis.append(np.mean(np.array(res)))

np.mean(train_train_simis)
np.mean(dev_dev_simis)
np.mean(test_test_simis)

def N_fold_split(N_folds, fold_ix, N_data):
    fold_ix = fold_ix % N_folds
    fold_size = np.int(N_data / N_folds)
    test_fold_start = fold_size * fold_ix
    test_fold_end   = fold_size * (fold_ix + 1)
    test_ixs  = range(test_fold_start, test_fold_end)
    train_ixs = list(range(0, test_fold_start)) + list(range(test_fold_end, N_data))
    return train_ixs, test_ixs

def read_standard_data(filename, mark):
    df = pd.read_table(filename, header=None)
    df = df.sample(frac=1, random_state=321)
    df.columns = ["smiles", "substance_id"]
    if mark == "pos": df["label"] = 1
    elif mark == "neg": df["label"] = 0
    else: raise ValueError
        
    df = df.iloc[:, [1, 0, 2]]
    
    del_list = ["C00735", "C03159", "C03161", "C05131", "C01592"]
    df = df[~df.substance_id.isin(del_list)]

    return df

def N_fold_split_dataset(pos_filename, neg_filename):
    df_pos = read_standard_data(pos_filename, "pos")
    df_neg = read_standard_data(neg_filename, "neg")
    
    for fold_ix in range(10):
        train_ixs_pos, test_ixs_pos = N_fold_split(10, fold_ix, len(df_pos))
        train_ixs_neg, test_ixs_neg = N_fold_split(10, fold_ix, len(df_neg))
        
        df_pos_train = df_pos.iloc[train_ixs_pos, :]
        df_pos_test = df_pos.iloc[test_ixs_pos, :]
        
        df_neg_train = df_neg.iloc[train_ixs_neg, :]
        df_neg_test = df_neg.iloc[test_ixs_neg, :]
        
        df_train = pd.concat([df_pos_train, df_neg_train], axis=0)
        df_test = pd.concat([df_pos_test, df_neg_test], axis=0)
        
        df_train = df_train.sample(frac=1, random_state=321)
        df_test = df_test.sample(frac=1, random_state=321)
        
        df_train["smiles"].to_csv("data/temp_train.csv", index=None)
        df_test["smiles"].to_csv("data/temp_test.csv", index=None)
        train_smi = Chem.SmilesMolSupplier("data/temp_train.csv")
        test_smi = Chem.SmilesMolSupplier("data/temp_test.csv")
        train_smi_fp = []
        for i in range(len(train_smi)):
            train_smi_fp.append(FingerprintMols.FingerprintMol(train_smi[i]))
        test_smi_fp = []
        for i in range(len(test_smi)):
            test_smi_fp.append(FingerprintMols.FingerprintMol(test_smi[i]))
        #test vs train
        test_train_simis = []
        for i in range(len(test_smi_fp)):
            res = []
            for j in range(len(train_smi_fp)):
                res.append(DataStructs.FingerprintSimilarity(test_smi_fp[i],train_smi_fp[j]))   
            test_train_simis.append(np.mean(np.array(res)))
        print("average similarity", np.mean(test_train_simis))
        
        
        # train 8768, test 973
        df_all = pd.concat([df_train, df_test], axis=0)
        print(df_all["label"].count())
        df_all.to_csv("data/folds/data_fold"+str(fold_ix)+".csv", index=None)

N_fold_split_dataset("data/Unstable3304.smi", "data/Stable6442.smi")


import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
#accept= pd.read_csv('experiencepce.csv')

train = pd.read_csv('H:/qdf\qdf\dataset/transf//train0.csv')
train.columns=['index','SMILES','PCE']
test = pd.read_csv('H:/qdf\qdf\dataset/transf//test0.csv')
test .columns=['index','SMILES','PCE']
val=  pd.read_csv('H:/qdf\qdf\dataset/transf/val0.csv')
val.columns=['index','SMILES','PCE']
smiles = train['SMILES']
property = train['PCE']

f = open ('H:/qdf\qdf\dataset/transf/train00.txt','w')

for ind, m in enumerate(smiles):
    print(ind)
       
    index=train['index'].iloc[ind]
    try:
        mol= Chem.MolToSmiles(Chem.MolFromSmiles(m))
        m3d=(Chem.MolFromSmiles(mol))
        E = AllChem.EmbedMolecule(m3d,
                              useRandomCoords=False,
                              ignoreSmoothingFailures=True)
    except:
        print('erro'+str(index))
        continue

    if E == -1:
        try:
            E = AllChem.EmbedMolecule(m3d,
                                  useRandomCoords=True,
                                  ignoreSmoothingFailures=True)
            AllChem.MMFFOptimizeMoleculeConfs(m3d, maxIters=10000)
        except:
            print('erro'+str(index))
            continue
    
    
    Draw.MolToImage(m3d, size=(250,250))
    conformer = m3d.GetConformer()
    coordinates = conformer.GetPositions()
   
    atoms = m3d.GetAtoms()
    atomic_symbols = np.array([i.GetSymbol() for i in atoms])
    
    atomic_symbols.resize(len(atomic_symbols),1)
   
    position = np.hstack((atomic_symbols,coordinates))

    proper=str(property[index])
    f.write('pce_'+ str(index)+'\n')
    for i in range(position.shape[0]):
        f.write(str(position[i][0]) + '    ' + str(position[i][1]) + '    ' + str(position[i][2]) + '    ' + str(position[i][3]) + '\n')
    f.write(proper+'\r\n')
f.close()

np.set_printoptions(threshold=sys.maxsize)
smilestest = test['SMILES']
propertest = test['PCE']
f = open ('H:/qdf\qdf\dataset/transf/test00.txt','w')
for ind, m in enumerate(smilestest):
    print(ind)
    index=test['index'].iloc[ind]
    try:
        mol= Chem.MolToSmiles(Chem.MolFromSmiles(m))
        m3d=(Chem.MolFromSmiles(mol))
        E = AllChem.EmbedMolecule(m3d,
                              useRandomCoords=False,
                              ignoreSmoothingFailures=True)
    except:
        print('erro'+str(index))
        continue

    if E == -1:
        try:
            E = AllChem.EmbedMolecule(m3d,
                                  useRandomCoords=True,
                                  ignoreSmoothingFailures=True)
            AllChem.MMFFOptimizeMoleculeConfs(m3d, maxIters=10000)
        except:
            print('erro'+str(index))
            continue
    
    Draw.MolToImage(m3d, size=(250,250))
    conformer = m3d.GetConformer()
    coordinates = conformer.GetPositions()

    atoms = m3d.GetAtoms()
    atomic_symbols = np.array([i.GetSymbol() for i in atoms])
    atomic_symbols.resize(len(atomic_symbols),1)
    position = np.hstack((atomic_symbols,coordinates))
    proper=str(propertest[index])
    f.write('pce_'+ str(index)+'\n')
    for i in range(position.shape[0]):
        f.write(str(position[i][0]) + '    ' + str(position[i][1]) + '    ' + str(position[i][2]) + '    ' + str(position[i][3]) + '\n')
    f.write(proper+'\r\n')
f.close()
np.set_printoptions(threshold=sys.maxsize)
smilesval = val['SMILES']
properval = val['PCE']
f = open ('H:/qdf\qdf\dataset/transf/val00.txt','w')
for ind, m in enumerate(smilesval):
    print(ind)
    index=val['index'].iloc[ind]
    try:
        mol= Chem.MolToSmiles(Chem.MolFromSmiles(m))
        m3d=(Chem.MolFromSmiles(mol))
        E = AllChem.EmbedMolecule(m3d,
                              useRandomCoords=False,
                              ignoreSmoothingFailures=True)
    except:
        print('erro'+str(index))
        continue

    if E == -1:
        try:
            E = AllChem.EmbedMolecule(m3d,
                                  useRandomCoords=True,
                                  ignoreSmoothingFailures=True)
            AllChem.MMFFOptimizeMoleculeConfs(m3d, maxIters=10000)
        except:
            print('erro'+str(index))
            continue
        
    
    Draw.MolToImage(m3d, size=(250,250))
    conformer = m3d.GetConformer()
    coordinates = conformer.GetPositions()

    atoms = m3d.GetAtoms()
    atomic_symbols = np.array([i.GetSymbol() for i in atoms])
    atomic_symbols.resize(len(atomic_symbols),1)
    position = np.hstack((atomic_symbols,coordinates))
    proper=str(properval[index])
    f.write('pce_'+ str(index)+'\n')
    for i in range(position.shape[0]):
        f.write(str(position[i][0]) + '    ' + str(position[i][1]) + '    ' + str(position[i][2]) + '    ' + str(position[i][3]) + '\n')
    f.write(proper+'\r\n')
f.close()
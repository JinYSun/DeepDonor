# QDF-Donor

QDF-Donor： Quantum deep field and transfer learning-assisted power conversion efficiency prediction for high-performance donor materials in organic solar cells

![图片](https://user-images.githubusercontent.com/62410732/150963296-aafd5845-c434-4137-9807-35040f1c43ce.png)

# Motivation

The QDF-Donor containing QDF-SM and QDF-P can predict the PCE of small molecule and polymer donor materials separately. 
 The small molecule and polymer donor materials  was collected from literature. Atoms and their 3D coordinates were used to represent the molecules after conformer optimization. The QDF-SM was trained on SM dataset. Gaussian-type orbital (GTO) was used to generate atomic basis function, and the molecular orbital was calculated by linear combination of atomic orbitals. The molecular orbital was corrected by computing external potential with a feed-forward DNN. The corrected molecular orbital was used to predict PCE with a DNN. The trained QDF-SM model was used to train the QDF-P on polymer molecule dataset materials by transfer learning with the same process of small molecule materials.
 
 ![图片](https://user-images.githubusercontent.com/62410732/150966587-07d32f3e-1f71-4dd8-bdf9-e1bb7fee2f88.png)


 
 # Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)

# Discussion

The [Discussion](https://github.com/JinyuSun-csu/QDF-Donor/tree/main/discussion) folder contains the scripts for evaluating the classification performance.  We compared sevaral common methods widely used in drug-likeness prediction, such as [GNN](https://github.com/JinyuSun-csu/QDF-Donor/blob/main/discussion/GNN.py),[RF](https://github.com/JinyuSun-csu/QDF-Donor/blob/main/discussion/RF.py), [ANN](https://github.com/JinyuSun-csu/QDF-Donor/blob/main/discussion/ANN.py),[GB](https://github.com/JinyuSun-csu/QDF-Donor/blob/main/discussion/GB.py).

# Usage

  If you want to make the prediction of PCE of unknown donor molecule, please put the molecule's SMILES files in to data directory and run [QDF-SM](https://github.com/JinyuSun-csu/QDF-Donor/blob/main/model/QDF_SM.py) for small molecules or [QDF-P](https://github.com/JinyuSun-csu/QDF-Donor/blob/main/model/QDF_P.py).

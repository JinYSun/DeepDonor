# DeepDonor

DeepDonor：Computational discovery of donor materials with high power conversion efficiency for organic solar cells

<img src="https://github.com/JinYSun/DeepDonor/blob/main/cover.jpg" alt="cover" style="zoom:25%;" />



# Motivation

The DeepDonor containing QDF-SM and QDF-PM can predict the PCE of small molecule and polymer donor materials separately. 
 The small molecule and polymer donor materials  was collected from literature. Atoms and their 3D coordinates were used to represent the molecules after conformer optimization. The QDF-SM was trained on SM dataset. Gaussian-type orbital (GTO) was used to generate atomic basis function, and the molecular orbital was calculated by linear combination of atomic orbitals. The molecular orbital was corrected by computing external potential with a feed-forward DNN. The corrected molecular orbital was used to predict PCE with a DNN. The trained QDF-SM model was used to train the QDF-P on polymer molecule dataset materials by transfer learning with the same process of small molecule materials.

# Data

The small molecule (SM) and polymer monomer (PM) datasets were built by collating existing experimental datasets and supplementing with a collection of high-PCE materials reported in the literature up to the end of 2021 separately. It can be used to train models for developing OSC donor materials. All the model were trained, validated and tested on the same dataset.

# Depends

We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).

[Anaconda](https://www.anaconda.com/)

[pytorch](https://pytorch.org/)

[rdkit](https://rdkit.org/)

By using the *environment.yml* file, it will install all the required packages.

```
`git clone https://github.com/jinysun/deepdonor.git
`cd DeepDonor`
`conda env create -f environment.yml`
`conda activate deepdonor`
```



# Discussion

The [Discussion](https://github.com/JinYSun/DeepDonor/tree/main/discussion) folder contains the scripts for evaluating the classification performance.  We compared sevaral common methods widely used in drug-likeness prediction, such as [GNN](https://github.com/JinYSun/DeepDonor/blob/main/discussion/GNN.py),[RF](https://github.com/JinYSun/DeepDonor/blob/main/discussion/RF.py), [ANN](https://github.com/JinYSun/DeepDonor/blob/main/discussion/ANN.py),[GB](https://github.com/JinYSun/DeepDonor/blob/main/discussion/GB.py).

# Usage

  If you want to make the prediction of PCE of unknown donor molecule, please put the molecule's SMILES files in to data directory and run [QDF-SM](https://github.com/JinYSun/DeepDonor/blob/main/model/QDF_SM.py) for small molecules or [QDF-PM](https://github.com/JinYSun/DeepDonor/blob/main/model/QDF_P.py) for polymer molecules.

```
  bash SM.sh/PM.sh for training
  bash predict.sh  for predicting
```



# Contact

Jinyu Sun E-mail: [jinyusun@csu.edu.cn](mailto:jinyusun@csu.edu.cn)

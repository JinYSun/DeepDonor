# DeepChemStable
DeepChemStable: chemical stability prediction using attention-based graph convolution network

## Environment
Python 3.6.3 </br>
Tensorflow 1.1.0 </br>
RDkit 2018.03.4 </br>
Autograd 1.2 </br>
Numpy 1.14.2 </br>
Pandas 0.23.4 </br>

## Model
The trained model weights are stored in `fingerprint_variables.bin`, `prediction_variables.bin`. </br>
Use the `predict.py` to predict. </br>
The predictions are saved in the file `results.csv`. </br>
The visualization of predictive unstable compounds with hightlighted unstable fragment are saved in the folder `figures/`. </br>

## Usage
`>python predict.py *yourfilepath* *amount*` </br>
Examples: `>python predict.py test.csv 8` </br>

Data file format: </br>
&nbsp;&nbsp;&nbsp;&nbsp;Datafile should be CSV file; </br>
&nbsp;&nbsp;&nbsp;&nbsp;The header must be "substance_id, smiles, label"; </br>
&nbsp;&nbsp;&nbsp;&nbsp;In "label" column, use "0" for all compounds so the visualization can implement. </br>

## Acknowledgement
The code is based on https://github.com/momeara/DeepSEA

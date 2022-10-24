#!/usr/bin/env python3

import argparse
import pickle
import sys

import torch

sys.path.append('../')
from train import train


if __name__ == "__main__":
    dataset_predict='ldxc'
    basis_set='6-31G'
    radius=0.75
    grid_interval=0.3
    
    # Setting of a neural network architecture.
    dim=250  # To improve performance, enlarge the dimensions.
    layer_functional=4
    hidden_HK=250
    layer_HK=3
    
    # Operation for final layer.
    #operation='sum'  # For energy (i.e., a property proportional to the molecular size).
    operation='sum'  # For homo and lumo (i.e., a property unrelated to the molecular size or the unit is e.g., eV/atom).
    
    # Setting of optimization.
    batch_size=2
    lr=8e-5
    lr_decay=0.8
    step_size=15
    iteration=100
    
    # num_workers=0
    num_workers=0
   
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dir_trained = '../dataset/' +  'transf/'
    dir_predict = '../dataset/' + dataset_predict + '/'

    field = '_'.join([basis_set, str(radius) + 'sphere', str(grid_interval) + 'grid/'])
    dataset_test = train.MyDataset(dir_predict + 'ldxc3_' + field)
    dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    with open(dir_trained + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals =256

    N_output = len(dataset_test[0][-2][0])

    model = train.QuantumDeepField(device, N_orbitals,
                                   dim, layer_functional, operation, N_output,
                                   hidden_HK, layer_HK).to(device)
    model.load_state_dict(torch.load('H:/library/QDF-Donor/model/trained/pdf_p.h5'
                                     ,
                                     map_location=device))
    tester = train.Tester(model)



    MAE, prediction = tester.test(dataloader_test, time=True)
    
    filename = ('../output/prediction--' +'.txt')
    tester.save_prediction(prediction, filename)

    print('results:', prediction)

    print('The prediction has finished.')

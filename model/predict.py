# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:46:29 2022

@author: Jinyu Sun
"""


import argparse
import pickle
import sys

import torch

sys.path.append('../')
from train import train


if __name__ == "__main__":
      """Args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius')
    parser.add_argument('grid_interval')
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_functional', type=int)
    parser.add_argument('hidden_HK', type=int)
    parser.add_argument('layer_HK', type=int)
    parser.add_argument('operation')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('step_size', type=int)
    parser.add_argument('iteration', type=int)
    parser.add_argument('molecule_type ')
    parser.add_argument('num_workers', type=int)
    args = parser.parse_args()
    dataset_predict = args.dataset
    unit = '(' + dataset.split('_')[-1] + ')'
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dim = args.dim
    layer_functional = args.layer_functional
    hidden_HK = args.hidden_HK
    layer_HK = args.layer_HK
    operation = args.operation
    batch_size = args.batch_size
    lr = args.lr
    lr_decay = args.lr_decay
    step_size = args.step_size
    iteration = args.iteration
    molecule_type  = args.molecule_type 
    num_workers = args.num_workers
   

   
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if molecule_type  = 'SM':
        dir_trained = '../dataset/' +  'SM/'
    else:
        dir_trained = '../dataset/' +  'PM/'
    dir_predict = '../dataset/' + dataset_predict + '/'

    field = '_'.join([basis_set, str(radius) + 'sphere', str(grid_interval) + 'grid/'])
    dataset_test = train.MyDataset(dir_predict + dataset + field)
    dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    with open(dir_trained + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals =256

    N_output = len(dataset_test[0][-2][0])

    model = train.QuantumDeepField(device, N_orbitals,
                                   dim, layer_functional, operation, N_output,
                                   hidden_HK, layer_HK).to(device)
    if molecule_type  = 'SM':
        model.load_state_dict(torch.load('../model/trained/SM.h5'
                                     ,
                                     map_location=device))
    else:
        model.load_state_dict(torch.load('../model/trained/PM.h5'
                                     ,
                                     map_location=device))
    tester = train.Tester(model)



    MAE, prediction = tester.test(dataloader_test, time=True)
    
    filename = ('../output/prediction--' +'.txt')
    tester.save_prediction(prediction, filename)


    print('results:', prediction)

    print('The prediction has finished.')

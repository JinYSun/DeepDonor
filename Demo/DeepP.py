#!/usr/bin/env python3

import argparse
import pickle
import sys

import timeit

import torch

sys.path.append('../')
from QDF_SM import train


if __name__ == "__main__":

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
    parser.add_argument('setting')
    parser.add_argument('num_workers', type=int)
    parser.add_argument('predataset')
    args = parser.parse_args()
    dataset = args.dataset
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
    setting = args.setting
    num_workers = args.num_workers
    predataset=args.predataset
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    """Create the dataloaders of training, val, and test set."""
   
    dir_dataset = '../dataset/' + dataset + '/'
    #field = '_'.join([basis_set, str(radius) + 'sphere', str(grid_interval) + 'grid/'])
    field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    dataset_train =train. MyDataset(dir_dataset + 'train_' + field)

   
    dataloader_train = train.mydataloader(dataset_train, batch_size, num_workers,
                                    shuffle=True)    
    
    
    dataset_test = train.MyDataset(dir_dataset  + 'test_' + field)
    dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    N_output = len(dataset_test[0][-2][0])

    model = train.QuantumDeepField(device, N_orbitals,
                                   dim, layer_functional, operation, N_output,
                                   hidden_HK, layer_HK).to(device)
    model.load_state_dict(torch.load('../output/model--' + predataset+'+'+basis_set,
                                     map_location=device))
    for para in model.parameters():
        para.requires_grad = True 
    trainer = train.Trainer(model, lr, lr_decay, step_size)
    tester = train.Tester(model)


    """Output files."""
    file_result = '../output/result--' + setting + '.txt'
    result = ('Epoch\tTime(sec)\tLoss_E\tLoss_V\tlosses_Et\tlosses_Vt\tMAE_train\t'
              'MAE_val' + unit + '\tMAE_test' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = '../output/prediction--' + setting + '.txt'
    file_model = '../output/model--' + setting

    print('Start training of the QDF model with', dataset, 'dataset.\n'
          'The training result is displayed in this terminal every epoch.\n'
          'The result, prediction, and trained model '
          'are saved in the output directory.\n'
          'Wait for a while...')

    start = timeit.default_timer()

    for epoch in range(iteration):
        loss_E, loss_V = trainer.train(dataloader_train)
        MAE_train=tester.test(dataloader_train)[0]
        MAE_test, prediction ,losses_Et,losses_Vt = tester.test(dataloader_test)
        time = timeit.default_timer() - start

        if epoch == 0:
            minutes = iteration * time / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*50)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_E, loss_V,losses_Et,losses_Vt,MAE_train,
                                   MAE_test]))
        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        tester.save_model(model, file_model)
        print(result)

    print('The training has finished.')

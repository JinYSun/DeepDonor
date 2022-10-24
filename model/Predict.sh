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
    operation='sum'  #  
    molecule_type = 'SM'      #'SM' for small molecules and 'PM' for polymer molecules
    # Setting of optimization.
    batch_size=2
    lr=8e-5
    lr_decay=0.8
    step_size=15
    iteration=100
    
    # num_workers=0
    num_workers=0
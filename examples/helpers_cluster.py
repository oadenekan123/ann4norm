#%% importing things
import numpy as np
import os
from ann4brains.nets import BrainNetCNN, load_model

working_dir = '/scratch/oadenekan/project_norm/ann4brains/examples'
# working_dir = r'C:\Users\oyina\src\senior_2019-2020\lab\bijsterbosch\project\oyin'
os.chdir(working_dir)

#%% reading in text files with patient data

def read_data(site_file):
    # open the file
    with open(site_file, 'r') as f:
        site_persons_data = [line.split('\n') for line in f]

    # create a list: each list is one person from site
    site_persons_data = [list_[0] for list_ in site_persons_data]
    site_persons_data = [patient.split('\t') for patient in site_persons_data]

    return site_persons_data

#%% turning site data into connectomes

def list_to_connectome(person_data, num_regions):

    # initialize the matrix
    matrix = np.ones((num_regions, num_regions))

    # upper triangualr side of matrix
    counter = 0
    for row in range(num_regions):
        for col in range(row+1, num_regions):
            matrix[row][col] = person_data[counter]
            counter = counter + 1
            # print("row:", row)
            # print("col", col)

    # print("sep")
    # lower triangular side of matrix
    counter = 0
    for col in range(num_regions):
        for row in range(col+1, num_regions):
            # matrix[row][]
            matrix[row][col] = person_data[counter]
            counter = counter + 1
            # print("col", col) 
            # print("row:", row)
    return np.array(matrix)

#%% connectomes to discriminator data

def connectomes_to_data(site_B_data, site_H_data, num_regions):
    # initialize array in which to hold site data; i is for channel dimension
    site_B_connectomes = np.ones((len(site_B_data), 1, num_regions, num_regions))
    site_H_connectomes = np.ones((len(site_H_data), 1, num_regions, num_regions))

    # create data matrices: each person call function
    for person in range(len(site_B_data)):
        site_B_connectomes[person, :, :, :] = list_to_connectome(site_B_data[person], num_regions)
    for person in range(len(site_H_data)):
        site_H_connectomes[person, :, :, :] = list_to_connectome(site_B_data[person], num_regions)

        
    # create y data
    # site b is first col, site h is second
    both_site_length = len(site_B_data) + len(site_H_data)
    y = np.zeros((both_site_length, 1))
    y[0:len(site_B_data), 0] = 1 # site b is first column
    y[len(site_B_data)+1:len(y), 0] = 0 # site h is second column

    # concatenate x data
    x = np.concatenate((site_B_connectomes, site_H_connectomes), axis=0)
    print(x.shape)
    print(y.shape)
    return (x,y)

#%% connectomes to generator data

def connectomes_to_generator_data(site_B_data, site_H_data, num_regions):
    # initialize array in which to hold site data; i is for channel dimension
    site_B_connectomes = np.ones((len(site_B_data), 1, num_regions, num_regions))
    site_H_connectomes = np.ones((len(site_H_data), 1, num_regions, num_regions))

    # create data matrices: each person call function
    for person in range(len(site_B_data)):
        site_B_connectomes[person, :, :, :] = list_to_connectome(site_B_data[person], num_regions)
    for person in range(len(site_H_data)):
        site_H_connectomes[person, :, :, :] = list_to_connectome(site_B_data[person], num_regions)
        
    # add option for which site to normalize to

    # concatenate x data
    x = site_B_connectomes
    y = site_H_connectomes
    num_nodes = num_regions * num_regions
    y = np.reshape(y, (len(site_H_connectomes), num_nodes))
    print(x.shape)
    print(y.shape)
    return (x,y)


#%% create e2n net
def e2n(net_name, h, w, n_injuries, max_iter = 10000, test_interval = 50, snapshot = 1000):
    # Specify the architecture using a list of dictionaries.
    e2n_arch = [
        ['e2n', # e2n layer 
        {'n_filters': 130, # 130 feature maps 
        'kernel_h': h, 'kernel_w': w  # Cross filter of size h x 1 by 1 x w (non-sliding, only on diagonal)
        }
        ], 
        ['dropout', {'dropout_ratio': 0.5}], # Dropout with 0.5 dropout rate.
        ['relu',    {'negative_slope': 0.33}], # Very leaky ReLU.
        ['fc',      {'n_filters': 30}],  # Fully connected/dense (Node-to-Graph when after e2n) layer
        ['relu',    {'negative_slope': 0.33}], # Very leaky ReLU
        ['out',     {'n_filters': n_injuries}] # Output with 2 nodes that correspond to 2 injuries.
    ]

    # Create BrainNetCNN model
    E2Nnet_sml = BrainNetCNN(net_name, # Unique model name.
                            e2n_arch, # List of dictionaries specifying the architecture.
                            hardware='cpu', # Or 'cpu'.
                            dir_data='./generated_synthetic_data', # Where to write the data to.
                            )
    #set pars
    E2Nnet_sml.pars['max_iter'] = max_iter # Train the model for 1000 iterations. (note this should be run for much longer!)
    E2Nnet_sml.pars['test_interval'] = test_interval # Check the valid data every 50 iterations.
    E2Nnet_sml.pars['snapshot'] = snapshot # Save the model weights every 1000 iterations.

    return E2Nnet_sml

#%% create e2e + e2n net
def e2e(net_name, h, w, n_injuries, max_iter = 10000, test_interval = 100, snapshot = 1000):
    # Specify the architecture.
    e2e_arch = [
        ['e2e', # e2e layer 
         {'n_filters': 32, # 32 feature maps 
          'kernel_h': h, 'kernel_w': w  # Sliding cross filter of size h x 1 by 1 x w
         }
        ], 
        ['e2n', {'n_filters': 64, 'kernel_h': h, 'kernel_w': w}],
        ['dropout', {'dropout_ratio': 0.5}],
        ['relu',    {'negative_slope': 0.33}],
        ['fc',      {'n_filters': 30}],
        ['relu',    {'negative_slope': 0.33}],
        ['out',     {'n_filters': n_injuries}] 
    ]

    # Create BrainNetCNN model
    E2Enet_sml = BrainNetCNN(net_name, e2e_arch, 
                             hardware='cpu', # Or 'cpu'.
                             dir_data='./generated_synthetic_data', # Where to write the data to.
                            )

    # Overwrite default parameters.
    # ann4brains.nets.get_default_hyper_params() shows the hyper-parameters that can be overwritten.
    #E2Enet_sml.pars['max_iter'] = 100000 # Train the model for 100K iterations.
    #E2Enet_sml.pars['test_interval'] = 500 # Check the valid data every 500 iterations.
    #E2Enet_sml.pars['snapshot'] = 10000 # Save the model weights every 10000 iterations.

    # NOTE using the above parameters takes awhile for the model to train (~2 hours ish? on a GPU)
    # If you want to do some simple fast experiments to start, use these settings instead.
    E2Enet_sml.pars['max_iter'] = max_iter # Train the model for 100 iterations (note this should be run for much longer!)
    E2Enet_sml.pars['test_interval'] = test_interval # Check the valid data every 20 iterations.
    E2Enet_sml.pars['snapshot'] = snapshot # Save the model weights every 100 iterations.

    return E2Enet_sml


#%% create e2n net
def e2n_generator(net_name, h, w, n_injuries, max_iter = 10000, test_interval = 50, snapshot = 1000):
    # Specify the architecture using a list of dictionaries.
    e2n_arch = [
        ['e2n', # e2n layer 
        {'n_filters': 130, # 130 feature maps 
        'kernel_h': h, 'kernel_w': w  # Cross filter of size h x 1 by 1 x w (non-sliding, only on diagonal)
        }
        ], 
        ['dropout', {'dropout_ratio': 0.5}], # Dropout with 0.5 dropout rate.
        ['relu',    {'negative_slope': 0.33}], # Very leaky ReLU.
        ['fc',      {'n_filters': 30}],  # Fully connected/dense (Node-to-Graph when after e2n) layer
        ['relu',    {'negative_slope': 0.33}], # Very leaky ReLU
        ['out',     {'n_filters': n_injuries}] # Output with 2 nodes that correspond to 2 injuries.
    ]

    # Create BrainNetCNN model
    E2Nnet_sml = BrainNetCNN(net_name, # Unique model name.
                            e2n_arch, # List of dictionaries specifying the architecture.
                            hardware='cpu', # Or 'cpu'.
                            dir_data='./generated_synthetic_data', # Where to write the data to.
                            )
    #set pars
    E2Nnet_sml.pars['max_iter'] = max_iter # Train the model for 1000 iterations. (note this should be run for much longer!)
    E2Nnet_sml.pars['test_interval'] = test_interval # Check the valid data every 50 iterations.
    E2Nnet_sml.pars['snapshot'] = snapshot # Save the model weights every 1000 iterations.

    return E2Nnet_sml

#%% randomly assign data to train, val, and test

def split_data(x, y, train_thresh, val_thresh):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    # seed random number generator
    seed_num = 0
    np.random.seed(seed_num)
  

    for idx, example in enumerate(x):
        
        # generate random number
        split_prob = np.random.random()
        # print(split_prob)

        # train
        if split_prob >= 0 and split_prob < train_thresh:
            train_x.append(x[idx])
            train_y.append(y[idx])
        # val
        elif split_prob >= train_thresh and split_prob < train_thresh + val_thresh:
            val_x.append(x[idx])
            val_y.append(y[idx])
        # test
        else:
            test_x.append(x[idx])
            test_y.append(y[idx])

    x_train = np.array(train_x, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.float32)
    x_val = np.array(val_x, dtype=np.float32)
    y_val = np.array(val_y, dtype=np.float32)
    x_test = np.array(test_x, dtype=np.float32)
    y_test = np.array(test_y, dtype=np.float32)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test

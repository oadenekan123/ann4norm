#!/usr/bin/env python
# coding: utf-8

# In[27]:


# get all pkgs
import os
import sys
import numpy as np
import pickle
import caffe

# To import ann4brains if not installed.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets import BrainNetCNN, load_model


# In[28]:


#get FNET data
import helpers_cluster as help
import importlib


#%% updating help
help = importlib.reload(help)

#%% read in data from the two sites
siteB_file = "FNETs_siteB.txt"
siteH_file = "FNETs_siteH.txt" 
num_regions = 10

site_B_data = help.read_data(siteB_file)
site_H_data = help.read_data(siteH_file)

# initialize array in which to hold site data; i is for channel dimension
site_B_connectomes = np.ones((len(site_B_data), 1, num_regions, num_regions))
site_H_connectomes = np.ones((len(site_H_data), 1, num_regions, num_regions))

# create data matrices
for person in range(len(site_B_data)):
    site_B_connectomes[person, :, :, :] = help.list_to_connectome(site_B_data[person], num_regions)
for person in range(len(site_H_data)):
    site_H_connectomes[person, :, :, :] = help.list_to_connectome(site_B_data[person], num_regions)

    
#%% create y data
# site b is first col, site h is second
both_site_length = len(site_B_data) + len(site_H_data)
y = np.zeros((both_site_length, 1))
y[0:len(site_B_data), 0] = 1 # site b is first column
y[len(site_B_data)+1:len(y), 0] = 0 # site h is second column

# concatenate
x = np.concatenate((site_B_connectomes, site_H_connectomes), axis=0)
print(x.shape)
print(y.shape)


# In[29]:


##randomly assign to train, val, and test
# create empty lists ot hold data
train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []

# seed random number generator
seed_num = 0
np.random.seed(seed_num)

# create train, val, and test thresholds
train_thresh = 0.7
val_thresh = 0.2
test_thresh = 0.1

x_train, y_train, x_val, y_val, x_test, y_test = help.split_data(x, y, train_thresh, val_thresh)

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

#%% rename data to fit into model
x_train = np.array(train_x, dtype=np.float32)
y_train = np.array(train_y, dtype=np.float32)
x_val = np.array(val_x, dtype=np.float32)
y_val = np.array(val_y, dtype=np.float32)
x_test = np.array(test_x, dtype=np.float32)
y_test = np.array(test_y, dtype=np.float32)

#%% print shapes of data
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

# print("y_train", y_train)
# print("y_val", y_val)
# print("y_test", y_test)




# In[30]:


##set parameters
# Number of outputs.
n_injuries = 1 # NOTE: The synthetic code only works for 2 injuries right now.

# Spatial height and width of the data. 
h = x_train.shape[2]
w = x_train.shape[3]
print(h, w)


# In[31]:


# Unique name for the model
net_name = 'E2Nnet_sml'

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
E2Nnet_sml.pars['max_iter'] = 100 # Train the model for 1000 iterations. (note this should be run for much longer!)
E2Nnet_sml.pars['test_interval'] = 50 # Check the valid data every 50 iterations.
E2Nnet_sml.pars['snapshot'] = 20 # Save the model weights every 1000 iterations.


# In[32]:


# %%
# Train (optimize) the network.
# WARNING: If you have a high max_iter and no GPU, this could take awhile...
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
#caffe.set_device(0)
#caffe.set_mode_cpu()
E2Nnet_sml.fit(x_train, y_train, x_val, y_val)  # If no valid data, could put test data here.

print("mission completed!")

# %%
# Plot the training iterations vs. the training loss, the valid data mean-absolute-difference, 
# and the valid data correlation with predicted and true (y_vald) labels.
file_name = os.getcwd() + "/models/plot_metrics.png"
# file_name = os.path.join(os.getcwd(), name)
E2Nnet_sml.plot_iter_metrics(True, file_name)

# %%
# Predict labels of test data
preds = E2Nnet_sml.predict(x_val)
preds = np.reshape(preds, (len(preds), 1))


# %%
# Compute the metrics.
E2Nnet_sml.print_results(preds, y_val)
print("predictions raw", preds)
print("y_test", y_val)
preds_trans = np.zeros((preds.shape))
preds_trans[preds >=0.5] = 1
preds_trans[preds < 0.5] = 0
print("predictions", preds_trans)

accuracy = np.sum(np.sum(y_val != preds_trans))
print("accuracy", accuracy)


# %%
# We can save the model like this.
# test_data = (x_test, y_test)
# file_name = "models/test_data.pkl"
# with open(file_name, 'wb') as pkl_file:
#         pickle.dump(test_data, pkl_file, protocol = 2)
E2Nnet_sml.save('models/E2Nnet_sml.pkl')



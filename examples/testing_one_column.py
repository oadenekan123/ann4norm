# get all pkgs
import os
import sys
import numpy as np
import pickle
import caffe
import random
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split


# To import ann4brains if not installed.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets import BrainNetCNN, load_model


# import helpers directory and update
import helpers_cluster as help
import importlib


#%% updating help
help = importlib.reload(help)

#%% create training data sets
siteB_file = "FNETs_siteB.txt"
siteH_file = "FNETs_siteH.txt" 
num_regions = 10

# read data from files
site_B_data = help.read_data(siteB_file)
site_H_data = help.read_data(siteH_file)

# create training data
x,y = help.connectomes_to_data(site_B_data, site_H_data, num_regions)

#%% split data 
# create train, val, and test thresholds
train_thresh = 0.7
val_thresh = 0.2
test_thresh = 0.1

x_train, y_train, x_val, y_val, x_test, y_test = help.split_data(x, y, train_thresh, val_thresh)

#%% set parameters and build model
# Number of outputs.
n_injuries = 1
h = x.shape[2]
w = x.shape[3]
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

net_name = 'E2Nnet_sml'
net = help.e2n(net_name, h, w, n_injuries)


#%% train model
net.fit(x_train, y_train, x_val, y_val)  # If no valid data, could put test data here.
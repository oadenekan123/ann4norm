#!/usr/bin/env python
# coding: utf-8

# In[27]:


# get all pkgs
import os
import sys
import numpy as np
import pickle
import caffe
import random
from datetime import datetime
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split


# To import ann4brains if not installed.
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append('/scratch/oadenekan/project_norm/ann4brains')
import ann4brains
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

#%% combine training and validaiton data for k-fold cross validation
x_train_val = np.concatenate((x_train, x_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)

#%% set up for splitting into k train and val
seed = 7
num_splits = 10

# randomly order indices of train_val data bank
random_order_idxs = list(range(x_train_val.shape[0]))
random.shuffle(random_order_idxs)

# split all data into pieces
splits = [] # create empty list to hold slices in
totalToDistr = x_train_val.shape[0]
minForEachSlice = int(totalToDistr/float(num_splits))
distrAtTop = totalToDistr % num_splits
distrAtTopCounter = 0
distMin = 0
for split in range(num_splits):
    distMax = distMin + minForEachSlice
    if distrAtTopCounter < distrAtTop:
        distMax = distMax + 1
        distrAtTopCounter = distrAtTopCounter + 1
    splits.append(random_order_idxs[distMin:distMax])


#%% train model

n_injuries = 1
h = x.shape[2]
w = x.shape[3]
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

net_name = 'E2Nnet_sml'
net = help.e2n(net_name, h, w, n_injuries)

accuracies = np.zeros((num_splits))
actual = []
og_preds = []
predicted = []
date = datetime.now().strftime("%d_%m_%Y")
for counter, val_idxs in enumerate(splits):
    
    # get data
    train_idxs = list(set(random_order_idxs) - set(val_idxs))
    x_train = x_train_val[train_idxs]
    y_train = y_train_val[train_idxs]
    x_val = x_train_val[val_idxs]
    y_val = y_train_val[val_idxs]

    # train model
    net.fit(x_train, y_train, x_val, y_val)  # If no valid data, could put test data here.
    print("mission completed!")

    # plot and save error over iterations
    file_name = os.getcwd() + "/models/{}_plot_metrics_{}_{}.png".format(date, net_name, counter)
    net.plot_iter_metrics(True, file_name)

    # predict
    preds = net.predict(x_val)
    preds = np.reshape(preds, (len(preds), 1))

    # Compute the accuracy
    net.print_results(preds, y_val)
    # print("predictions raw", preds)
    # print("y_val", y_val)
    preds_trans = np.zeros((preds.shape))
    preds_trans[preds >=0.5] = 1
    preds_trans[preds < 0.5] = 0
    accuracy = np.mean(np.sum(y_val == preds_trans))
    print("accuracy", accuracy)

    # save accuracy, actual, and predicted
    accuracies[counter] = accuracy
    actual.append(y_val)
    og_preds.append(preds)
    predicted.append(preds_trans)

    # save the model
    net.save('models/()_{}_{}.pkl'.format(date, net_name, counter))
    counter = counter + 1

#%% look at accurary
print("accuracies", accuracies)
ave_accuracy = np.mean(accuracies)
print("average accuracy:", ave_accuracy)

#%% save data after training
data = (accuracies, actual, og_preds, predicted)
file_name = os.getcwd() + '/models/{}_{}_cross_val_iter_data.pkl'.format(date, net_name)
with open(file_name, 'wb') as pkl_file:
        pickle.dump(data, pkl_file, protocol = 2)

#%% importing things
import pickle
import numpy as np
import caffe

#%% read in model and data
file_name = "models/E2Nnet_sml.pkl"
with open(file_name, 'rb') as model_file:
        E2Nnet_sml = pickle.load(model_file)

file_name = "models/test_data.pkl"
with open(file_name, 'rb') as data_file:
        test_data = pickle.load(data_file)

x_test = test_data[0]
y_test = test_data[1]

# %% Predict labels of test data
preds = E2Nnet_sml.predict(x_test)


# %% Compute the metrics.
E2Nnet_sml.print_results(preds, y_test)


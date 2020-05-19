#%% importing things
import numpy as np
import os
working_dir = '/home/oadenekan/src/cgan_norm/oyin'
working_dir = r'C:\Users\oyina\src\senior_2019-2020\lab\bijsterbosch\project\oyin'
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

#%% importing things
import numpy as np
import os
working_dir = '/home/oadenekan/src/cgan_norm/oyin'
working_dir = r'C:\Users\oyina\src\senior_2019-2020\lab\bijsterbosch\project\oyin'
os.chdir(working_dir)

#%%
# i do not have the commands
# go through hisotry for this
# get private IP and enter into putty
# sudo yum install openssh-server
# curl ifconfig.me
# sudo apt-get
# sudo systemctl enable ssh
# sudo systemctl start sshd
# sudo systemctl status sshd
# sudo systemctl enable sshd
# curl ifconfig.me
# ifconfig
# 

#%% reading in text files
siteB = "FNETs_siteB.txt"
siteH = "FNETs_siteH.txt"

with open(siteB, 'r') as f:
    siteB = [line.split('\n') for line in f]

siteB = [list_[0] for list_ in siteB]
# print(siteB)
# siteB = [list_.split('\t') for list_ in siteB]
siteB = [patient.split('\t') for patient in siteB]

# type(siteB[0].split('\t'))
len(siteB[0])
len(siteB)
#%% creating matrix
siteB_pt1 = siteB[0]
num_regions = 10
num_patients = 115
matrix = np.ones((num_regions, num_regions))

sum = 0
for num in range(num_regions):
    sum = sum + num

counter = 0
for row in range(num_regions):
    for col in range(num_regions):
        if counter >= sum-1:
            counter = 0
        if row != col:
            matrix[row][col] = siteB_pt1[counter]
            counter = counter + 1

print(counter)
print(sum)
# %%
# print(matrix)
print(matrix[0][num_regions-1] == matrix[num_regions-1][0])
print(matrix[0])
print(matrix[9])


# %% creating matrix attempt part letter 2
# upper triangular part
test_dim = 5
num_regions = 10
counter = 0
for row in range(num_regions):
    for col in range(row+1, num_regions):
        matrix[row][col] = siteB_pt1[counter]
        counter = counter + 1
        # print("row:", row)
        # print("col", col)

print("sep")
# lower triangular matrix
counter = 0
for col in range(num_regions):
    for row in range(col+1, num_regions):
        # matrix[row][]
        matrix[row][col] = siteB_pt1[counter]
        counter = counter + 1
        # print("col", col) 
        # print("row:", row)


#%%
np.array(matrix).shape
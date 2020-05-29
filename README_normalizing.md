This repository is forks from the ann4brains repository and the code is repurposed to normalize resting-state fMRI data form different sites.
We use the idea of a cyclic-GAN to normalize the data, based on the following cycle-GAN paper: 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'.

We create a yml file to create a conda environment to run the code called 'proj_env.yml'. To create the environment, execute the command 'conda env create -n norm_env -f /ann4brains/proj_env.yml'. 
To activate the environment, execute the command 'export PATH=/export/Anaconda3-5.2.0/bin:$PATH' and then 'source activate ./norm_env/'. 

We use the code in the ann4brains repository to create a discriminator and generator. The models are built in the '/ann4brains/examples' directory.
There is a file containing helper functions for the repurposing of the repository called 'helpers_cluster.py'.

We built the discirmiantor based on the 'brainnetcnn.ipynb' example script. There are two networks contiand in this file.
We create the 'brain_net_cnn_e2e_corss_val.py' and 'brain_net_cnn_e2n_cross_val.py' to cross validate the models.

We started building the generator in 'brain_net_cnn_generator_test.py'. To create the generator, we using a modified version of the 
'/ann4brains/ann4brains/nets.py' script called 'ann4brains/ann4brains/nets_gen.py'. 'nets_gen.py' is imported at the beginning of the generator script.

There is still plenty of work to be done.
1) Get the generator working. I believe the key is to converting gen_output from a blob/tesnor to an array in the reconstruct_all_connectomes function in 'ann4brains/ann4brains/nets_gen.py' script. 
2) Thoroughly read the cyclic-GAN paper and ensure that the generator is correctly and fully implemented.
3) Allow for customizaiton of the cycle gan loss functions (allow for changing weights -- changing the amount each loss contrubtes to total GAN loss).
4) Assess the performance of the discriminator and the generator.
5) Assess the performance of the cyclic system.

Best of luck.

import os 
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import *

from utils.split_condition import split_conditions
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas


def compute_noise_ceilling(subj_list):
    """
    Computes the noise ceilling per voxel
    """
    for i, subj in enumerate(subj_list):
        train_path = os.path.join(betas_dir, f'{subj}_betas_list_nativesurface_train.npy')
        test_path = os.path.join(betas_dir, f'{subj}_betas_list_nativesurface_test.npy')
        betas_mask = np.load(os.path.join(betas_dir, f'{subj}_betas_list_nativesurface_train_test_mask.npy'), allow_pickle=True)
        if subj == 'subj06' or subj == 'subj08':
            roi_mask = os.path.join(mask_dir, subj, f'short.reduced.nans.{subj}.testrois.npy')
        else:
            roi_mask = os.path.join(mask_dir, subj, f'short.reduced.{subj}.testrois.npy')
        betas_train = np.load(train_path, allow_pickle=True)
        betas_test = np.load(test_path, allow_pickle=True)
        mask = np.load(roi_mask, allow_pickle=True)
        betas_train = betas_train[:,betas_mask]
        noise_ceilling_all_vox = np.zeros(betas_train.shape[0])
        print(noise_ceilling_all_vox.shape)
        noise_ceilling_file = os.path.join(noise_dir, f'{subj}_noise_ceilling_all_vox.npy')
        if os.path.exists(noise_ceilling_file):
            continue


        for row in range(betas_train.shape[0]):  
            if row % 1000 == 0: 
                print(f'------- FINDING NOISE CEILLING FOR ROW {row}') 
         
            current_row = np.array([betas_train[row]])
        

            new_row, _ = rescale(current_row, betas_test[row])
   
            if row == 0:
                print(new_row)
         
            rss = np.sum((new_row - betas_test[row])**2)   
    

            variance = np.sum((betas_test[row] - np.mean(betas_test[row]))**2)
            ve_voxels = 1 - rss / variance    # noise ceilling of all voxels in current roi 
            if row % 1000 == 0:
                print(f'-------  NOISE CEILLING FOR ROW {row} is {ve_voxels}') 
            noise_ceilling_all_vox[row] = ve_voxels

        np.save(noise_ceilling_file, noise_ceilling_all_vox)
    


def rescale(train, test):
    """
    Input 
    -------
    train: 1D array of length n, to rescale
    test : 1D array of length n, target to rescale to

    Output 
    --------
    new_train: 1D array of length n, rescale train array

    Rescale the train array to the test array. Linear rescale using pseudoinverse
    """
    if len(train.shape) == 1:
        train = np.array([train]) # add a dimension for concat
   # print(train.shape)
    train_ones = np.concatenate((train, np.ones((train.shape[0], train.shape[1])))).T
    scale = np.linalg.pinv(train_ones) @ test.T 

    new_train = train_ones @ scale  #make 
    return new_train.T.squeeze(), scale

compute_noise_ceilling(subj_list)

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:51:58 2023

@author: LuisWork
"""

import numpy as np



def split_conditions(data, conditions, splitting_conditions):
    lookup = np.unique(conditions)
    n_conds = lookup.shape[0]
    n_conds_split = np.unique(splitting_conditions).shape[0]
    n_dims = data.ndim
    n_voxels, n_betas = data.shape
    
    # will store the data here
    if n_dims == 2:
        n_voxels, _ = data.shape
        train_data = np.empty((n_voxels, n_conds)).astype(np.float32)
        test_data = np.empty((n_voxels, n_conds_split)).astype(np.float32)
    else:
        x, y, z, _ = data.shape
        train_data = np.empty((x, y, z, n_conds)).astype(np.float32)
        test_data = np.empty((x, y, z, n_conds_split)).astype(np.float32)
        
    test_indexes = np.zeros(n_conds).astype(bool)
    test_conditions = []
    train = 0
    test = 0
    for j, x in enumerate(lookup):
        conds_bool = conditions == x
        if x not in splitting_conditions:
            # simply add to train
            if n_dims == 2:
                    train_data[:, train] = np.nanmean(data[:, conds_bool][:, 0], axis = 1)
            else:
                    train_data[:, :, :, train] = np.nanmean(data[:, :, :, conds_bool][:, :, :, 0], axis = 3)
        else:
            # put M-1 trials in training set and average, keep the last trial in the test set. Add condition id to the list
            if n_dims == 2:
                train_data[:, train] = np.nanmean(data[:, conds_bool][:, :-1], axis = 1)
                test_data[:, test] = data[:, conds_bool][:, -1]
            else:
                train_data[:, :, :, train] = np.nanmean(data[:, :, :, conds_bool][:, :-1], axis = 3)
                test_data = data[:, :, :, conds_bool][:, -1]
            test_indexes[train] = True
            test_conditions.append(x)
            test += 1
        train += 1
    return train_data, test_data, test_indexes, test_conditions


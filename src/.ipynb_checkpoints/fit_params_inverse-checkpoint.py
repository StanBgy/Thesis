import os
from re import T

from pandas.plotting import parallel_coordinates
import numpy as np
from numpy._typing import _16Bit
from scipy.optimize import curve_fit
import pandas as pd
import math
import time 
from utils.utils import *
from utils.rf_gaussians import gaussian_2d_curve, gaussian_2d_curve_pol


"""
Here, we fit the gaussian curve to each voxel 
"""

def gaussian_fit(subj_list, rois, params,  mode='averaged'):
    targetspace = 'nativesurface'
    columns = ["x0", "y0", "sigma", "slope", "intercept"]
    initial = params['initial']
    bounds = params['bounds']
    for i, subj in enumerate(subj_list):
        print(f'GETTING BETAS FOR SUBJ0{i+1}')
        betas_file = os.path.join(betas_dir , f'{subj}_betas_list_{targetspace}_{mode}.npy') # could parametertize the targetsurface but eh
        betas = np.load(betas_file, allow_pickle=False).astype(np.float32).T
        if np.isnan(betas).any():
            print(f'FOUND NANS in subj0{i+1} s betas')

        if mode == "train":
            betas_test_file = os.path.join(betas_dir, f'{subj}_betas_list_{targetspace}_test.npy')
            print(f'LOADING TEST BETAS FOR SUBJ0{i+1}')
            betas_test = np.load(betas_test_file, allow_pickle=False).astype(np.float32).T

            train_test_mask_file = os.path.join(betas_dir, f'{subj}_betas_list_{targetspace}_train_test_mask.npy')
            print(f'LOADING TRAIN TEST MSAK FOR SUBJ0{i+1}')
            train_test_mask = np.load(train_test_mask_file, allow_pickle=False).T.astype(bool)


        print(f'Starting fitting for subj0{i + 1}')
        n_betas, n_voxels = betas.shape

        model_allROI = np.zeros((len(rois), n_voxels, len(columns)))
        start = time.time()

        for voxel in range(n_voxels):
            voxel_activity = betas[:, voxel]

            for i, roi in enumerate(list(rois.keys())):
                mds_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_MDS_rotated_VO-1_{mode}.npy')  # we only care about rotate
                mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

                fit_file = os.path.join(fits_dir, 'fits_inversed', subj, f'fits_{subj}_{mode}_{roi}_inversed.npy')
                    
                if os.path.exists(fit_file):
                    print(f'a fitted model already exists for {roi}, already exist \n Delete it if refitting \
                          or fitting new voxels is needed ')
                    continue   # This is not ideal. But I dont want to create a model for each voxel 


                if params['random']:
                    attempt = 1
                    solved = False
                    while not solved and attempt <= 10:
                        try:  
                            initial_guess = (initial[1] - initial[0]) * np.random.random(initial[0].shape) + initial[0]
                            voxel_fit = curve_fit(
                                    gaussian_2d_curve_pol,
                                    mds,
                                    voxel_activity,
                                    p0 = initial_guess,
                                    bounds = bounds,
                                    method = 'trf',
                                    ftol = 1e-06,
                            )[0]
                            solved = True
                        
                        except RuntimeError:
                            print(f'VOXEL {voxel}: optimal params not found after {attempt} attempts')
                            attempt + 1
            
                model_allROI[i, voxel] = voxel_fit # put the fitted values on the rigth voxel to ROI index 
            if not voxel % 100:
                print(f'\t\tFitted Voxel {voxel} out of {n_voxels}, elapsed time on {subj}: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}' )
        for i, roi in enumerate(list(rois.keys())):
            fit_file = os.path.join(fits_dir, 'fits_inversed', subj, f'fits_{subj}_{mode}_{roi}_inversed.npy')
                    
            if os.path.exists(fit_file):
                print(f'a fitted model already exists for {roi}, skipping')
                continue
                
            fits_roi = pd.DataFrame(model_allROI[i], columns=columns)

            mds_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_MDS_rotated_VO-1_train.npy')  # Works better if I dont forget this  
            mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T
            
            def gaus_roi(fits):
                return gaussian_2d_curve_pol(mds, *fits)

            pred_activity = fits_roi.apply(gaus_roi, axis = 1)
            pred_activity = np.array([np.array(x) for x in pred_activity]).T
            roi_res = np.sum((pred_activity - betas)**2, axis=0)
            roi_tot = sum((betas- np.tile(betas.mean(axis=0), (n_betas, 1)))**2).T

            
            if mode == "train":
                mds_test = mds[:, train_test_mask]
                def gaus_roi_test(fits):
                    return gaussian_2d_curve_pol(mds_test, *fits)
                pred_activity_test = fits_roi.apply(gaus_roi_test, axis=1)
                pred_activity_test = np.array([np.array(x) for x in pred_activity_test]).T
                roi_res_test = np.sum((pred_activity_test - betas_test)**2, axis=0)
                roi_rot_test = sum((betas_test - np.tile(betas_test.mean(axis=0), (sum(train_test_mask), 1)))**2).T
                fits_roi["test_var_explained"] = 1 - roi_res_test / roi_rot_test
            fits_roi["var_explained"] = 1 - roi_res / roi_tot
            fits_roi["mds_ecc"] = (fits_roi.x0 ** 2 + fits_roi.y0[1] ** 2) ** (1/2)
            fits_roi["mds_ang"] = np.arctan2(fits_roi.x0/bounds[1][0], fits_roi.y0/bounds[1][1])

            np.save(fit_file, fits_roi)
            print(f'file for {roi} has been saved ')
    del betas

gaussian_fit(subj_list, rois, params,  mode="train")

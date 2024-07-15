import os
import nsd_access 
import numpy as np 
import pandas as pd
from utils.utils import *
from utils.rf_gaussians import gaussian_2d_curve, gaussian_2d_curve_pol

"""
To avoid overloading the RAM, the fits for the full brain models were stored after computation, and the variance expain (and the other variables)
wasn't computed. This script does exactly that
"""

def variance_full_brain(subj_list, rois):
    columns = ["x0", "y0", "sigma", "slope", "intercept"]
    targetspace = 'nativesurface'
    mode = 'train'
    bounds = params['bounds']
    for i, subj in enumerate(subj_list):
        print(f'------ENTERING {subj} -------')
        
        
        for k in range(5):

            betas_file = os.path.join(nsd_dir, 'full_brain', subj  , f'{subj}_betas_list_{targetspace}_{mode}_full_{k}.npy') # could parametertize the targetsurface but eh
            betas_chunk = np.load(betas_file, allow_pickle=True, mmap_mode='r').astype(np.float32).T  

            betas_test_file = os.path.join(nsd_dir, 'full_brain', subj  , f'{subj}_betas_list_{targetspace}_test_full_{k}.npy')
            print(f'LOADING TEST BETAS FOR {subj}')
            betas_test = np.load(betas_test_file, allow_pickle=True, mmap_mode='r').astype(np.float32).T
    
            train_test_mask_file = os.path.join(nsd_dir, 'full_brain', subj, f'{subj}_betas_list_{targetspace}_train_test_mask_full_{k}.npy')
            print(f'LOADING TRAIN TEST MSAK FOR {subj}')
            train_test_mask = np.load(train_test_mask_file, allow_pickle=True, mmap_mode='r').T.astype(bool)
            
            fits_raw_file = os.path.join(fits_dir, 'fits_inversed', subj, f'fits_{subj}_train_full_{k}_raw.npy')
            if not os.path.exists(fits_raw_file):
                print(f"----- FITS DO NOT EXISTS FOR {subj} on chunk {k} !!!!!!")
                continue
        
            fits_raw = np.load(fits_raw_file,  mmap_mode='r')
            for j, roi in enumerate(list(rois.keys())):
            

                
                    
                final_fits_file = os.path.join(fits_dir, 'fits_inversed', subj, f'fits_{subj}_train_{roi}_full_{k}.npy')
                if not os.path.exists(final_fits_file):
                    fits_roi = pd.DataFrame(fits_raw[j], columns=columns)

                    mds_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_MDS_rotated_VO-1_train.npy')  # we only care about rotate
                    mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

                    def gaus_roi(fits):
                        return gaussian_2d_curve_pol(mds, *fits)
        
                    pred_activity = fits_roi.apply(gaus_roi, axis = 1)
                    pred_activity = np.array([np.array(x) for x in pred_activity]).T
                    roi_res = np.sum((pred_activity - betas_chunk)**2, axis=0)
                    roi_tot = sum((betas_chunk- np.tile(betas_chunk.mean(axis=0), (betas_chunk.shape[0], 1)))**2).T
                  
                    
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
        
                    np.save(final_fits_file, fits_roi)
                    print(f'file for {roi} has been saved ')
                    del fits_roi

                   
                    
                else:
                    print(f'File already exists for {roi} at chunk {k}')

variance_full_brain(subj_list, rois)
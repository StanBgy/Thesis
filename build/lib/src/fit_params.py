import os
from re import T
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

def gaussian_fit(subj_list, rois, params, rotated=False, mode='averaged'):

    columns = ["x0", "y0", "sigma", "slope", "intercept"]
    initial = params['initial']
    bounds = params['bounds']

    for i, subj in enumerate(subj_list):
        print(f'GETTING BETAS FOR SUBJ0{i+1}')
        betas_file = os.path.join(betas_dir , f'{subj}_betas_list_nativesurface_{mode}.npy')
        betas = np.load(betas_file, allow_pickle=False).astype(np.float32).T
        print(f'Starting fitting for subj0{i}')
        n_betas, n_voxels = betas.shape
        for roi in rois.keys():
            start = time.time()
            if rotated:
                mds_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_mds_{mode}.npy')
            if not rotated:
                mds_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_MDS_rotated_VO-2_{mode}.npy')

            mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

            if rotated:
                fit_file = os.path.join(fits_dir, 'fits_rotated', f'fits_{subj}_{mode}_{roi}_rotated.npy')
            if not rotated:
                fit_file = os.path.join(fits_dir, 'fits_rotated', f'fits_{subj}_{mode}_{roi}_notrotated.npy')
            
            if os.path.exists(fit_file):
                print(f'\t\t\ŧskipping {roi}, already exists')
                continue
            fits_roi = pd.DataFrame(columns=columns)

            print(f'Starting fitting for ROI {roi}, on version 1')
            for voxel in range(n_voxels):
                if not voxel % 1000:
                    print(f'\t\t\t\t\t[{100*voxel/n_voxels:.2f}%] elapsed time since {roi} start: ',
                            f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}'
                         )
                voxel_activity = betas[:, voxel]

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
                                    ftol = 1e-06
                            )[0]
                            solved = True
                        
                        except RuntimeError:
                            print(f'VOXEL {voxel}: optimal params not found after {attempt} attempts')
                            attempt + 1

                fits_roi.loc[voxel] = voxel_fit

            def gaus_roi(fits):
                return gaussian_2d_curve_pol(mds, *fits)

            pred_activity = fits_roi.apply(gaus_roi, axis = 1)
            pred_activity = np.array([np.array(x) for x in pred_activity]).T
            roi_res = np.sum((pred_activity - betas)**2, axis=0)
            roi_tot = sum((betas- np.tile(betas.mean(axis=0), (n_betas, 1)))**2).T

             ## Maybe include split here

            fits_roi["var_explained"] = 1 - roi_res / roi_tot
            fits_roi["mds_ecc"] = (fits_roi.x0 ** 2 + fits_roi.y0[1] ** 2) ** (1/2)
            fits_roi["mds_ang"] = np.arctan2(fits_roi.x0/bounds[1][0], fits_roi.y0/bounds[1][1])

        np.save(fit_file, fits_roi)
        print(f'\t\t\tTime elapsed during {roi}: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}'
                    )
    del betas



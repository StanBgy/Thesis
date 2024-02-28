import os
import time
import pandas as pd
import numpy as np
from utils.utils import *


def create_models(subj_list, sior, rois, models, mode='averaged', rotated=False):
    """
    Take the fitted betas and create a model 
    If split is train we also add the cross validated var_explained to the model
    Then we use these models to get them into the cortical surface """
    for sub in subj_list:
        start_sub = time.time()
        print(f'Enter {sub} at {time.strftime("%H:%M:%S", time.gmtime(start_sub))}')
        maskdata_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy')
        maskdata = np.load(maskdata_file, allow_pickle=True).astype(int)

        belongs = []
        for i in maskdata:
            belongs.append(sior[i])

        if mode == 'averaged':
            columns = ["x0", "y0", "sigma", "slope", "intercept",  "var_explained", "mds_ecc", "mds_ang"]
        if mode == 'train':
            columns = ["x0", "y0", "sigma", "slope", "intercept", 'test_var_explained', "var_explained", "mds_ecc", "mds_ang"]

        models_files = {}
        all_exist = True
        for m in models:
            model_file = os.path.join(models_dir, f'best_fits_{m}_{sub}_{mode}_basevoxel.npy')
            if not os.path.exists(model_file):
                model_out = build_model(m)
                for roi in rois.keys():
                    print(f'{roi}: {model_out.loc[model_out == roi]}')
                np.save(model_file, model_out)
                print(f'\t\t\t{m} saved to disk')
            else:
                print(f'\t\t\t{m} already exists')
            

def build_model(model, columns):
    model_out = pd.DataFrame(np.zeros((n_voxels, len(columns))), columns=columns, dtype="Float32")
    model_out.var_explained = -np.inf
    model_out["roi"] = belongs
    model_out["fit_with"] = -1
    for roi_name, roi_value in rois.items():
        if not rotated:
            fits_ss_file = os.path.join(
                fits_dir, 'fits_not_rotated', sub, f'fits_{sub}_{mode}_{roi_name}_notrotated.npy'
                )
        if rotated:
            fits_ss_file = os.path.join(
                    fits_dir, 'fits_rotated', sub, f'fits_{sub}_{mode}_{roi_name}_rotated.npy'
                    )
        fits_ss = pd.DataFrame(np.load(fits_ss_file, allow_pickle=True), columns=columns, dtype = "Float32")
        fits_ss["roi"] = belongs   
        fits_ss["fit_with"] = roi_name


        match model:
            case 'wself':
                fit_mask = model_out.var_explained < fits_ss.var_explained
            case 'oself':
                only_self = fits_ss.roi == fits_ss.fit_with
                fit_mask = model_out.var_explained < fits_ss.var_explained
                fit_mask = np.logical_and(only_self, fit_mask)
        if any(fit_mask):
            update_fits = fits_ss[fit_mask]
            model_out.update(update_fits)
        print(model_out.roi)
    return model_out

import os
import time
import pandas as pd
import numpy as np
from utils.utils import *


"""
Not in use right now (07/05) 
"""

def create_models_best(subj_list, sior, rois, models, mode='train', rotated=True):
    """
    Take the fitted betas and create a model 
    If split is train we also add the cross validated var_explained to the model
    Then we use these models to get them into the cortical surface """
    for sub in subj_list:
        start_sub = time.time()
        print(f'Enter {sub} at {time.strftime("%H:%M:%S", time.gmtime(start_sub))}')
        if sub == 'subj06' or sub == 'subj08':
            maskdata_file = os.path.join(mask_dir, sub, f'short.reduced.nans.{sub}.testrois.npy')
        else:
            maskdata_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy')
        maskdata = np.load(maskdata_file, allow_pickle=True).astype(int)
        n_voxels = maskdata.shape[0]

        belongs = []
        for i in maskdata:
            belongs.append(sior[i])

        columns = ["x0", "y0", "sigma", "slope", "intercept", "test_var_explained", "var_explained", "mds_ecc", "mds_ang"]

        models_files = {}
        all_exist = True
        for m in models:
            if rotated:
                model_file = os.path.join(models_dir, f'best_fits_{m}_{sub}_{mode}_bestroi.npy')
            if not os.path.exists(model_file):
                model_out = build_model(m, columns, n_voxels, belongs, sub, mode, rotated)
                for roi in rois.keys():
                    print(model_out)
                    print(f'{roi}: {model_out.loc[model_out.roi == roi]}')
                np.save(model_file, model_out)
                print(f'\t\t\t{m} saved to disk')
            else:
                print(f'\t\t\t{m} already exists')
            

def build_model(model, columns, n_voxels, belongs, sub, mode='train', rotated=True): 
    model_out = pd.DataFrame(np.zeros((n_voxels, len(columns))), columns=columns, dtype="Float32")
    model_out.var_explained = -np.inf
    model_out["roi"] = belongs
    model_out["fit_with"] = -1
    for roi_name, roi_value in rois.items():
        if rotated:
            fits_ss_file = os.path.join(
                    fits_dir, 'fits_inversed', sub, f'fits_{sub}_{mode}_{roi_name}_inversed.npy'
                    )
        fits_ss = pd.DataFrame(np.load(fits_ss_file, allow_pickle=True), columns=columns, dtype = "Float32")
        fits_ss["roi"] = belongs   
        fits_ss["fit_with"] = roi_name


        match model:
            case 'wself':
                fit_mask = model_out.var_explained < fits_ss.var_explained
            case 'woself':
                skip_woself = fits_ss.roi != fits_ss.fit_with
                fit_mask = model_out.var_explained < fits_ss.var_explained
                fit_mask = np.logical_and(skip_woself, fit_mask)
            case 'oself':
                only_self = fits_ss.roi == fits_ss.fit_with
                fit_mask = model_out.var_explained < fits_ss.var_explained
                fit_mask = np.logical_and(only_self, fit_mask)
        if any(fit_mask):
            update_fits = fits_ss[fit_mask]
            model_out.update(update_fits)

        model_out['fit_with'] = model_out.groupby(['roi'])['var_explained'].mean().sort_values(ascending=False).index[0]
    return model_out

models = ['woself']
create_models_best(subj_list, sior, rois, models)


import os
import time
import pandas as pd
import numpy as np
from utils.utils import *


"""
Modified version of the create_model file to accomodate for the "bestROI" model: 
We use the result matrix, collapse it over hemi, and then for each subj we know which ROI is better explained by which 

Then, the "best_roi" model takes that into account and for each ROI, only save the paramters of the ROI that is the best sampling space 
This leads us with a model that only takes fitted values from the overall best sampling space 
"""

def create_models(subj_list, sior, rois, models, mode='train', rotated=True):
    """
    Take the fitted betas and create a model 
    If split is train we also add the cross validated var_explained to the model
    Then we use these models to get them into the cortical surface """
    all_results_file = os.path.join(proj_dir, 'results', 'results_bestROI_hemis_collapsed.npy')
    all_results = np.load(all_results_file, allow_pickle=True)
    results_all_subj = np.mean(all_results, axis=0)

    columns = ["x0", "y0", "sigma", "slope", "intercept", "test_var_explained", "var_explained", "mds_ecc", "mds_ang"]
    results_df = pd.DataFrame(results_all_subj, columns=rois.keys(), index=rois.keys(), dtype=float)
    results_df['best_roi'] = results_df.idxmax(axis=1)

    # create a dict that stores the maskdata value and the best roi use the result_df we computed above.
    # Convintent to tell our model where to look (which roi has the best result when sample for which roi)
    sior_best_roi = {roi_value: best_roi for roi_value, best_roi in zip(sior.keys(), results_df['best_roi'])}
    
    
    for i, sub in enumerate(subj_list):
        start_sub = time.time()
        print(f'Enter {sub} at {time.strftime("%H:%M:%S", time.gmtime(start_sub))}')
        if sub == 'subj06' or sub == 'subj08':
            maskdata_file = os.path.join(mask_dir, sub, f'short.reduced.nans.{sub}.testrois.npy')
        else:
            maskdata_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy')
        maskdata = np.load(maskdata_file, allow_pickle=True).astype(int)
        n_voxels = maskdata.shape[0]

        belongs = []
        for j in maskdata:
            belongs.append(sior[j])

        # now we can put in a list, for each index, which one is the best 
        best_roi_list = []
        for k in maskdata:
            best_roi_list.append(sior_best_roi[k])

        
        models_files = {}
        all_exist = True
        for m in models:
            if rotated:
                model_file = os.path.join(models_dir, f'best_fits_{m}_{sub}_{mode}.npy')
            if not os.path.exists(model_file) or os.path.exists(model_file):
                model_out = build_model(m, columns, n_voxels, belongs, best_roi_list, sub, mode, rotated)
                for roi in rois.keys():
                    print(model_out)
                    print(f'{roi}: {model_out.loc[model_out.roi == roi]}')
                np.save(model_file, model_out)
                print(f'\t\t\t{m} saved to disk')
            else:
                print(f'\t\t\t{m} already exists')
            

def build_model(model, columns, n_voxels, belongs, best_roi_list, sub, mode='train', rotated=True): 
    model_out = pd.DataFrame(np.zeros((n_voxels, len(columns))), columns=columns, dtype="Float32")
    model_out.var_explained = -np.inf
    model_out["roi"] = belongs
    model_out['fit_with'] = best_roi_list  #right? so then I just update when this corresponds to roi name
    
    ### load all and collapse?? (maybe I should add that to the original fit script)
                    ### I dont think I have do to that 
                    ### Only update model_out when results['bestROI'] is equal to current roi 
    ### then check the resutst, and do the thing 
    for roi_name, roi_value in rois.items():
        if rotated:
            fits_ss_file = os.path.join(  # so this is one each voxel is doing when sampling form V1
                    fits_dir, 'fits_inversed', sub, f'fits_{sub}_{mode}_{roi_name}_inversed.npy'
                    )
        fits_ss = pd.DataFrame(np.load(fits_ss_file, allow_pickle=True), columns=columns, dtype = "Float32")
        fits_ss["roi"] = belongs   
        fits_ss['fit_with'] = roi_name

        match model:
            case 'best_roi':
                find_best = model_out.fit_with == fits_ss.fit_with 
                fit_mask = model_out.var_explained < fits_ss.var_explained
                fit_mask = np.logical_and(find_best, find_best)
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

    return model_out

#models = ['best_roi']
#create_models_best(subj_list, sior, rois, models)

default_path = os.path.expanduser('~')

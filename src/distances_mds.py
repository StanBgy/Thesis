import os 
import math
import numpy as np
import pandas as pd
from utils.utils import *

models_subs = {}
models = ['oself']

for i, sub in enumerate(subj_list):
    models_subs[sub] = {}
    for m in models:
        m_file = os.path.join(models_dir, f'best_fits_{m}_{sub}_train_basevoxel_rotated.npy')
        models_subs[sub][m] = pd.DataFrame(np.load(m_file, allow_pickle=True), columns=columns)
        models_subs[sub][m][columns[:-2]] = models_subs[sub][m][columns[:-2]].astype(np.float32)

for sub in models_subs.keys():
    print(f'ENTERING {sub}')
    for hemi in hemis:
        print(f'Working on hemi {hemi}')
        model_dic = models_subs[sub][models[0]]
        if sub == 'subj06' or sub == 'sub08':
            maskdata_file = os.path.join(mask_dir, sub, f'{hemi}.short.{sub}.testrois.npy') # CHANGE THIS LATER 
            # I NEED TO FIX THIS MASK TOO 
        maskdata_file = os.path.join(mask_dir, sub, f'{hemi}.short.{sub}.testrois.npy')
        maskdata = np.load(maskdata_file, allow_pickle=True).astype(int)
        if hemi == 'lh':
            model_hemi = model_dic[:maskdata.shape[0]]
        if hemi == 'rh':
            model_hemi = model_dic[maskdata.shape[0]:]
        for roi_name, roi_value in rois_distances.items():
            roi_distances_file = os.path.join(proj_dir, 'distances', sub, f'{hemi}.{sub}.dists.{roi_name}.npy')
            if not os.path.exists(roi_distances_file):
                print(f'\t\tcomputing distances for {roi_name}')
                model_roi = model_hemi.loc[[x in rois_distances[roi_name] for x in maskdata]]
                pairs = list(zip(model_roi.x0, model_roi.y0))
                ppairs = [(a, b) for idx, a in enumerate(pairs) for b in pairs]
                voxel1_ind = np.repeat(np.array(range(1, len(pairs) + 1)), len(pairs))
                voxel2_ind = np.tile(np.array(range(1, len(pairs) + 1)), len(pairs))
                ms_roi_distances = pd.DataFrame(ppairs, columns=['voxel1', 'voxel2'])
                ms_roi_distances['voxel1_ind'] = voxel1_ind
                ms_roi_distances['voxel2_ind'] = voxel2_ind
                ms_roi_distances['distance'] = ms_roi_distances.apply(
                    lambda x: math.sqrt((x.voxel1[0] - x.voxel2[0]) ** 2 + (x.voxel1[1] - x.voxel2[1]) ** 2), axis=1)
                np.save(roi_distances_file, ms_roi_distances.distance)
                print(f'\t\tsaving distances for {roi_name}')
            else:
                print(f'\t\tskipping {roi_name} since distances file already exists')

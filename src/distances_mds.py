import os 
import math
import numpy as np
import pandas as pd
from utils.utils import *
from utils.flips import get_prefered_xy



def compute_distance(subj_list, rois, sessions, models, hemis) -> None: 
    """

    Compute the distances, direction and differences between the fitted positions of every pairs of voxels 
    
    ------ Inputs -----

    subj_list: list of subjects

    rois: dict of ROIS and their keys

    sessions: list of number of sessions per subject, needed for the cos and sin function

    models: list of models we want (here, best_roi and oself)

    hemis: just ["lh", "rh"]

    -------- Outputs ------

    None, everything is saved as .npy
    """

    models_subs = {}

    cos_sin = get_prefered_xy(subj_list, sessions, fetch_conds=False)


    for i, sub in enumerate(subj_list):
        models_subs[sub] = {}
        for m in models:
            if m != 'oself':
                m_file = os.path.join(models_dir, f'best_fits_{m}_{sub}_train.npy')  #I should rename these 
            else:
                m_file = os.path.join(models_dir, f'best_fits_{m}_{sub}_train_basevoxel_rotated.npy')
            models_subs[sub][m] = pd.DataFrame(np.load(m_file, allow_pickle=True), columns=columns)
            models_subs[sub][m][columns[:-2]] = models_subs[sub][m][columns[:-2]].astype(np.float32)
            models_subs[sub][m]['x0_prefered'] = models_subs[sub][m]['x0'] * cos_sin[i, 0]-models_subs[sub][m]['y0'] * cos_sin[i, 1]
            models_subs[sub][m]['y0_prefered'] = models_subs[sub][m]['x0'] * cos_sin[i, 1]+models_subs[sub][m]['y0'] * cos_sin[i, 0]

    for i, sub in enumerate(models_subs.keys()):
        print(f'ENTERING {sub}')
        for hemi in hemis:
            print(f'Working on hemi {hemi}')
            for model in models: 
                model_dic = models_subs[sub][model]
                if sub == 'subj06' or sub == 'subj08':
                    maskdata_file = os.path.join(mask_dir, sub, f'{hemi}.short.nans.{sub}.testrois.npy') 
                    # mask fixed - only the cortical surface is messed up 
                else:
                    maskdata_file = os.path.join(mask_dir, sub, f'{hemi}.short.{sub}.testrois.npy')
                maskdata = np.load(maskdata_file, allow_pickle=True).astype(int)
                if hemi == 'lh':
                    model_hemi = model_dic.iloc[:maskdata.shape[0]]
                    print(model_hemi.shape)
                if hemi == 'rh':
                    model_hemi = model_dic.iloc[-maskdata.shape[0]:]
                    print(model_hemi.shape)# unsure about this 
                for roi_name, roi_value in rois_distances.items():
                    roi_aligned_file = os.path.join(proj_dir, 'distances', sub, f'{hemi}.{model}.{sub}.xyaligned.{roi_name}.npy')
                    roi_difference_file = os.path.join(proj_dir, 'distances', sub, f'{hemi}.{model}.{sub}.difference.{roi_name}.npy')
                    roi_distances_file = os.path.join(proj_dir, 'distances', sub, f'{hemi}.{model}.{sub}.dists.{roi_name}.npy')
                    roi_direction_file = os.path.join(proj_dir, 'distances', sub, f'{hemi}.{model}.{sub}.directions.{roi_name}.npy')
                    if not os.path.exists(roi_distances_file) or not os.path.exists(roi_direction_file) or not os.path.exists(roi_difference_file):
                        # or not os.path.exists(roi_direction_file):
                        print(f'\t\tcomputing distances and direction for {roi_name}')
                        model_roi = model_hemi.loc[[x in rois_distances[roi_name] for x in maskdata]]
                        old_pairs = list(zip(model_roi.x0, model_roi.y0))
                        pairs = list(zip(model_roi.x0_prefered, model_roi.y0_prefered))
                        old_ppairs = [(a, b) for idx, a in enumerate(old_pairs) for b in old_pairs]
                        ppairs = [(a, b) for idx, a in enumerate(pairs) for b in pairs]
                        voxel1_ind = np.repeat(np.array(range(1, len(pairs) + 1)), len(pairs))
                        voxel2_ind = np.tile(np.array(range(1, len(pairs) + 1)), len(pairs))
                        ms_roi_distances = pd.DataFrame(ppairs, columns=['voxel1', 'voxel2'])
                        ms_roi_distances['voxel1_ind'] = voxel1_ind
                        ms_roi_distances['voxel2_ind'] = voxel2_ind
                        ms_roi_distances['distance'] = ms_roi_distances.apply(
                            lambda x: math.sqrt((x.voxel1[0] - x.voxel2[0]) ** 2 + (x.voxel1[1] - x.voxel2[1]) ** 2), axis=1)
                        ms_roi_distances['directions'] = ms_roi_distances.apply(  # y2-y1; x2-x1 
                            lambda x: np.arctan2((x.voxel2[1] - x.voxel1[1]), (x.voxel2[0] - x.voxel1[0])) * (180 / np.pi), axis=1) # this is correct: voxel n is a x,y tuple
                        ms_roi_distances['difference'] = ms_roi_distances.apply(
                            lambda x: (x.voxel1[0] - x.voxel2[0], x.voxel1[1] - x.voxel2[1]) if  x.voxel1[0] >= x.voxel2[0] 
                                    else (x.voxel2[0] - x.voxel1[0], x.voxel2[1] - x.voxel1[1]), axis=1) 
                
                        np.save(roi_distances_file, ms_roi_distances.distance)
                        np.save(roi_direction_file, ms_roi_distances.directions)
                        np.save(roi_difference_file, ms_roi_distances.difference)
                        print(f'\t\tsaving distances, directions and differences for {roi_name}')
                    else:
                        print(f'\t\tskipping {roi_name} since distances file already exists')


#compute_distance(subj_list, rois, sessions, models, hemis)

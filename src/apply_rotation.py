import os
import pandas as pd
import numpy as np
from utils.flips import create_rotation_df, get_ranking
from utils.utils import *


def apply_rotation(target, subj, rois, mode='averaged'):
    """
    Take a target ROI and a subject: 
    rotated all the ROI's MDS based on the optimal rotation matrix
    to the given target (which in this case is the ROI with the lowest voerall distance to all others)

    This also managed flipped ROIS, so some of them might be flipped 
    I should add a print statment for that 
    """

    rotations_df = create_rotation_df(subj, rois, random=False, mode=mode)
    rotations_df = rotations_df[rotations_df['base'] == target]
    rotations_df = get_ranking(rotations_df, only_filter=True)
    rotations_df = rotations_df.reset_index(drop=True)
    rotations_df.to_csv(f'rotations/{subj}_rotations_df.csv', index=True)
    print(rotations_df)
    for roi in rois.keys():
        rotated_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_MDS_rotated_{target}_{mode}.npy')
        if not os.path.exists(rotated_file):
            source_mds_file = os.path.join(mds_dir, subj, f'{subj}_{roi}_mds_{mode}.npy')
            source_mds = np.load(source_mds_file, allow_pickle=True)
            if roi == target: 
                ### if the current roi is our target, just keep the MDS as is but rename it, for simplicity sake
                np.save(rotated_file, source_mds, allow_pickle=True)
                continue ### necessary 
            
            U = rotations_df.loc[rotations_df['source'] == roi, 'U'].squeeze()
            # I need to save this, somehow 
            print(U.shape)
            # flip first
            if "flipped" in rotations_df[rotations_df['source'] == roi].target[0]:
                source_mds = np.dot(source_mds, np.array([[-1, 0], [0, 1]]))
            rotated_mds = np.dot(source_mds, U)
            print(f"Saving Rotated mds for {subj} and ROI {roi}")
            np.save(rotated_file, rotated_mds)
        else: 
            print(f"Rotated mds for {subj} and ROI {roi} already exists!")







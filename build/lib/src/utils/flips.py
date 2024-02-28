import os 
import copy
import pandas as pd
import numpy as np
from utils.kabsch2D import * 
from utils.utils import *

"""
Utils functions for the flips 

create_rotation_df loads every mds and compare them against each others 
creates a df that has, for every ROI, the error compared to each others ROI, flipped and not flipped

take subj as an argument, and split, so we can use the splitted MDS or the averaged ones
rois is the dict of ROIS + index, useful for iterating over ROIS 


get_ranking takes a df as an argument, and returns its ranking, sorted, with the distance of each ROI compared to all others 
This manages for random as well 


get_rank_dict takes our list of subj, the rois and the split boolean as arugments, and return one dict with ranks as intergers, 
and another one with ranks and their actual values 

"""


def create_rotation_df(subj, rois, split=False, random=True):
    cols = ['source', 'base', 'target', 'U', 'distance']
    comparisions = (len(rois) * (len(rois)-1)) * 2 
    if random: comparisions += 1 
    rotations = np.zeros((comparisions, len(cols)), dtype=object)
    i = 0
    randoms = []
    x = 2 
    if random: 
        x +=1
    for roi_source in rois.keys():

        mds_source_file = os.path.join(mds_dir, subj, f'{subj}_40_{roi_source}_mds.npy') # REMEMBER TO RENAME THAT 
        mds_source = np.load(mds_source_file, allow_pickle=True)
        
        for roi_target in rois.keys():
            if roi_source != roi_target:

                mds_target_file = os.path.join(mds_dir, subj, f'{subj}_40_{roi_target}_mds.npy')
                mds_target = np.load(mds_target_file, allow_pickle=True)

                # there is probably a smarter way of doing that, but for now, it works
                for j in range(x): # 0 is normal, 1 flipped, 2 random
                    rotations[i, 1] = roi_target


                    if j == 1:
                        mds_target = np.dot(mds_target, np.array([[-1, 0], [0, 1]]))  # that little array should do the work 
                        roi_target = roi_target + "_flipped"

                    if j == 2:
                        # shuffle the mds by permuting their indexes 
                        permut_idx_source = np.random.permutation(len(mds_source))
                        permut_idx_target = np.random.permutation(len(mds_target))
                        shuffled_source = mds_source[permut_idx_source]
                        shuffled_target = mds_target[permut_idx_target]

                        # compute optimal rotation
                        U_shuffled, t_shuffled = kabsch2D(shuffled_source, shuffled_target, translate=True)
                        # based on that, rotate the source 
                        rotated_source = rotate(shuffled_source, U_shuffled)
                        # get the error between the rotated source and the target
                        randoms.append(avg_distance(rotated_source, shuffled_target, t_shuffled))
                        del rotated_source, shuffled_source, shuffled_target # just in case
                        continue # no need to do it everytime

                    #find optimal rotation
                    if random == 'only': # only return the distance between randomly flipped MDS; used for permutation
                        permut_idx_source = np.random.permutation(len(mds_source))
                        permut_idx_target = np.random.permutation(len(mds_target))
                        mds_source = mds_source[permut_idx_source]
                        mds_target = mds_target[permut_idx_target]

                    U, t = kabsch2D(mds_source, mds_target, translate=True)
                    rotations[i, 0] = roi_source
                    rotations[i, 2] = roi_target
                    rotations[i, 3] = U

                    # rotate based on U
                    rotated_source = rotate(mds_source, U)

                    # find the average distance
                    rotations[i, 4] = avg_distance(rotated_source, mds_target, t)

                    i += 1
    if random:

        rotations[i, 0] = "random"
        rotations[i, 1] = "random"
        rotations[i, 2] = "random"
        rotations[i, 4] = np.mean(randoms)

    df = pd.DataFrame(rotations, columns=cols)
    df = df.astype({'distance': 'float32'})  # needed for later

    return df 


def get_ranking(df, return_mean=False, only_filter=False):
    # get the index of the lowest error for each pair in each ROI
    # I also allow for only return the mean: useful for paramteric test 
    indx_to_keep = df.groupby(['source', 'base'])['distance'].idxmin()
    df_filtered = df.loc[indx_to_keep]
    if only_filter:
        return df_filtered
    if return_mean:
        rank = df_filtered.groupby('source')['distance'].mean()
        return rank
    rank = df_filtered.groupby('source')['distance'].mean().sort_values()
    return rank



def get_rank_dict(subj_list, rois, split=False, random=True):
    dict, dict_values = {}, {}
    for i in range(len(subj_list)):
        rotations_df = create_rotation_df(subj=subj_list[i], rois=rois, split=split, random=random)
      #  print(rotations_df)
        rank = get_ranking(rotations_df)
        if random:
            rois_with_random = copy.deepcopy(rois)
            rois_with_random['random'] = 13
            rank_int = [rois_with_random[roi] for roi in rank.index]
        else: 
            rank_int = [rois[roi] for roi in rank.index]
        dict[subj_list[i]] = rank_int
        dict_values[subj_list[i]] = rank

    return dict, dict_values


def get_distances(df, rois):
    """ 
    Not the right spot, but this takes a single subject df and returns a matrix of all get_distances
    need for Wilocoxon test """
    cols = list(rois.values())
    distances = np.zeros((len(cols), len(cols)), dtype=object)
    for roi_source, i in rois.items():
        for roi_target, j in rois.items():
            roi_distance = df.loc[(df["source"] == roi_source) & (df["base"] == roi_target)]["distance"].values
            if len(roi_distance) > 0:
                distances[i-1, j-1] = roi_distance[0]

    return distances

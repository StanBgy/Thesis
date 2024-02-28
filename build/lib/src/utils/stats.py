import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import random
from utils.utils import param_dir

from src.utils.flips import create_rotation_df, get_distances, get_ranking 

"""
Here I will store all my function necessary for statistical anaylsis, 
making the notebook clearers 

I will store different test for different applications, 
I might separate them if it gets messy"""


def rank_correlation(rank_dict):
    """
    Compute the correlation between ranks in a dictionnary
    which contains the ranking for all subjects, ordered with ROI-matching intergers 

    We use the kendall tau statsitc to compute the correlation between each pair
    """
    corr = np.zeros((len(rank_dict), len(rank_dict)))
    p_values = np.zeros((len(rank_dict), len(rank_dict)))
    
    for i in range(len(rank_dict)):
        for j in range(len(rank_dict)):
            res = stats.kendalltau(list(rank_dict.values())[i], list(rank_dict.values())[j])
            corr[i, j] = res.correlation
            p_values[i, j] = round(res.pvalue, 5)

    corr_df = pd.DataFrame(corr, columns=rank_dict.keys(), index=rank_dict.keys())
    p_values_df = pd.DataFrame(p_values, columns=rank_dict.keys(), index=rank_dict.keys())

    return corr_df, p_values_df


def parametric_test(subj_list, rois, iterations):
    for i in range(len(subj_list)):
        param_file_name = os.path.join(param_dir, f'{subj_list[i]}_parametric_test_output.csv')
        if not os.path.exists(param_file_name):
            zeros = np.zeros((iterations + 1, len(rois)))
            df_param = pd.DataFrame(zeros, columns=list(rois.keys()))

            df_true = create_rotation_df(subj_list[i], rois, random=False)
            means_true = get_ranking(df_true, return_mean=True)
            df_param.loc[0, :] = means_true

            for x in range(1, iterations + 1):
                df_it = create_rotation_df(subj_list[i], rois, random='only')
                means_it = get_ranking(df_it, return_mean=True)
                df_param.loc[x, :] = means_it

            df_param.to_csv(param_file_name, index=False)
    

def wilcoxon_test(subj_list, rois):
    """
    compute the array of all distances between all ROIS and all participants
    Then compute the T and P value between each ROI pairs by taking all participants' values"""
    distance = np.zeros((len(subj_list), len(rois), len(rois))) # big 3D matrix to store all distances in a manageable way 

    cols = rois.keys()
    ints = rois.values()

    T_values = np.zeros((len(cols), len(cols)), dtype=object)
    p_values = np.ones((len(cols), len(cols)),  dtype=object)
    T_values_df = pd.DataFrame(T_values, columns =cols, index=cols)
    p_values_df = pd.DataFrame(p_values, columns= cols, index=cols)

    for i, sub in enumerate(subj_list):
        distance_df = create_rotation_df(sub, rois, random=False)
        distance[i] = get_distances(distance_df, rois)

    for i in ints: 
        i -= 1
        for j in ints:
            j -= 1
            if i == j:
                continue # skip when it is the same, but I might want to remove that 

            x = np.delete(distance, (i, j), axis = 1)[:, :, i].flatten() # this grabs all the corresponding columns, delete the unwanted values 
            # being i and j, and then flatten it onto 1D
            y = np.delete(distance, (i, j), axis = 1)[:, :, j].flatten()

            res = stats.wilcoxon(x, y)
            T_values_df.iloc[i, j] = res.statistic
            p_values_df.iloc[i, j] = res.pvalue

    return T_values_df, p_values_df

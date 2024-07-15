import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import random
from utils.utils import param_dir

from utils.flips import * 

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
    """
    Create a 1000 random rotation matrix and compare it to the ouputed one. 
    """
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
    

def wilcoxon_test(subj_list, rois, mode='averaged'):
    """
    compute the array of all distances between all ROIS and all participants
    Then compute the T and P value between each ROI pairs by taking all participants' values"""
    distance = np.zeros((len(subj_list), len(rois), len(rois))) # big 3D matrix to store all distances in a manageable way 

    cols = rois.keys()
    ints = rois.values()

    z_values = np.zeros((len(cols), len(cols)), dtype=object)
    p_values = np.ones((len(cols), len(cols)),  dtype=object)
    medians = np.zeros((len(cols), len(cols)), dtype=object)
    z_values_df = pd.DataFrame(z_values, columns =cols, index=cols)
    p_values_df = pd.DataFrame(p_values, columns= cols, index=cols)
    median_df = pd.DataFrame(medians, columns=cols, index=cols)

    for i, sub in enumerate(subj_list):
        distance_df = create_rotation_df(sub, rois, random=False, mode=mode)
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

            # make the data a 1D array : X - Y
            x -= y
        
            
           
            # print median of the difference 
            # make a table o
            median_df.iloc[i, j] = np.median(x)  # fix the indexing 

            res = stats.wilcoxon(x, method='approx') # change that to output Z (approx) 
            ### compare it to matlab 
      #      print(res.zstatistic)
           
            z_values_df.iloc[i, j] = np.abs(res.zstatistic) * np.sign(np.median(x))
            p_values_df.iloc[i, j] = res.pvalue

    return z_values_df, p_values_df, median_df

def plot_colorcoded(df):
    """plot color coded tables of a df"""
    
    norm = plt.Normalize(df.values.min(), df.values.max())


    cmap = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axis('tight')
    ax.axis('off')
    

    cell_colors = cmap(norm(df.values))
    
 
    table = ax.table(cellText=df.values.round(2),
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellColours=cell_colors,
                     cellLoc='center',
                     loc='center')
    

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    
   
    plt.show()

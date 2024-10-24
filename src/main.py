from load_betas import load_betas
from noise_ceilling import compute_noise_ceilling
from create_rdm import create_rdm
from apply_rotation import apply_rotation
from fit_params import gaussian_fit
from fit_params_inverse import gaussian_fit_inverse   
from create_models import create_models
from distances_mds import compute_distance
from utils.utils import *


mode = "train"
rotated=True
full_brain = False 
distances = False  # Change to false if you dont care about the distance/correlation part of the analysis


if __name__ == "__main__":
    # First step : get the betas
    
    # we want the not full betas either way for RDM + MDS creation
    load_betas(subj_list, sessions, targetspace=targetspace, mode=mode)

    print("----- Betas done -----")

    compute_noise_ceilling(subj_list)
    
    print('---- Noise Ceilling done------')
    # Use the betas to create both RDM and MDS 
    create_rdm(subj_list, mode=mode)

    print('----- RDM and MDS done--------')

    # Apply Rotation 
    if rotated:
        apply_rotation('VO-1', subj, rois, mode=mode)

    print('-----Rotation done-------')

    # Fit on the gaussian curve per ROI 
    gaussian_fit(subj_list, rois, params, rotated=True, mode=mode) 

    print('-----Fit 1 done -----')

    # fit on the gaussian curve per voxel 
    gaussian_fit_inverse(subj_list, rois, params, mode=mode)

    print('----- Fit 2 done--------')

    # Create model 
    create_models(subj_list, sior, rois, models, mode=mode, rotated=True)
    # Export model stays at a notebook; I think it is better that way 
    
    print('------Model created--------')
    # compute the distances between all fitted voxels' prefered positions 
    if distances: 
        compute_distance(subj_list, rois, sessions, models, hemis)

        print('----distances done -------')



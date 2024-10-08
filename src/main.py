from load_betas import load_betas
from noise_ceilling import compute_noise_ceilling
from create_rdm import create_rdm
from apply_rotation import apply_rotation
from fit_params import gaussian_fit
from fit_params_inverse import gaussian_fit_inverse   
from create_models import create_models
from utils.utils import *


mode = "train"
rotated=True
full_brain = False 


if __name__ == "__main__":
    # First step : get the betas
    if full_brain: 
        load_betas_full(subj_list, sessions)
    
    # we want the not full betas either way for RDM + MDS creation
    load_betas(subj_list, sessions, targetspace=targetspace, mode=mode)

    compute_noise_ceilling(subj_list)

    # Use the betas to create both RDM and MDS 
    create_rdm(subj_list, mode=mode)

    # Apply Rotation 
    if rotated:
        apply_rotation('VO-1', subj, rois, mode=mode)

    # Fit on the gaussian curve per ROI 
    gaussian_fit(subj_list, rois, params, rotated=True, mode=mode) 

    # fit on the gaussian curve per voxel 
    gaussian_fit_inverse(subj_list, rois, params, mode=mode)

    # Create model 
    create_models(subj_list, sior, rois, models, mode=mode, rotated=True)
    # Export model stays at a notebook; I think it is better that way 


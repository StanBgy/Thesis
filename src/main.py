from load_betas import load_betas
from create_rdm import create_rdm
from apply_rotation import apply_rotation
from fit_params import gaussian_fit
from create_models import create_models
from utils.utils import *


mode = "train"
if __name__ == "__main__":
    # First step : get the betas
    load_betas(subj_list, sessions, targetspace=targetspace, mode=mode) # change the None here 

    # Use the betas to create both RDM and MDS 
    create_rdm(subj_list, mode=mode)

    # Apply Rotation 
    for subj in subj_list:
        apply_rotation('VO-1', subj, rois, mode=mode)

    # Fit on the gaussian curve
    gaussian_fit(subj_list, rois, params, rotated=True, mode=mode) 

    # Create model 
    create_models(subj_list, sior, rois, models, mode=mode, rotated=True)
    # Export model will stay as a notebook for now 


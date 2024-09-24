from load_betas import load_betas
from create_rdm import create_rdm
from apply_rotation import apply_rotation
from fit_params import gaussian_fit
from create_models import create_models
from create_models_bestroi import create_models_best
from utils.utils import *


mode = "train"
best_roi = True
rotated=True
full_brain = False 

if __name__ == "__main__":
    # First step : get the betas
    if full_brain: 
        load_betas_full(subj_list, sessions)
    
    # we want the not full betas either way for RDM + MDS creation
    load_betas(subj_list, sessions, targetspace=targetspace, mode=mode)

    # Use the betas to create both RDM and MDS 
    create_rdm(subj_list, mode=mode)

    # Apply Rotation 
    if rotated:
        apply_rotation('VO-1', subj, rois, mode=mode)

    # Fit on the gaussian curve
    gaussian_fit(subj_list, rois, params, rotated=True, mode=mode) 

    # Create model 
    if best_roi: 
        create_models_best(subj_list, sior, rois, models, mode=mode, rotated=True)
    else: 
        create_models(subj_list, sior, rois, models, mode=mode, rotated=True)
    # Export model will stay as a notebook for now 


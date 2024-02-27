from load_betas import load_betas
from create_rdm import create_rdm
from apply_rotation import apply_rotation
from fit_params import gaussian_fit
from utils.utils import *

# First step : get the betas
if __name__ == "__main__":
    print(subj_list)
    load_betas(subj_list, sessions, targetspace=targetspace, split=None) # change the None here 

# Use the betas to create both RDM and MDS 
    create_rdm(subj_list, split=False)

# Apply Rotation 
    for subj in subj_list:
        apply_rotation('VO-2', subj, mode='averaged')

# Fit on the gaussian curve
    gaussian_fit(subj_list, rois, params, split=False, rotated=True, mode='averaged') # Need to update to add split=True version 

# Create model 
#
# Export model 


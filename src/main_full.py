from load_betas_full import load_betas_full
from clean_betas import clean_betas
from fit_params_fullbrain import gaussian_fit_full
from variance_full_brain import variance_full_brain


mode = "train" # defaulted as train in all functions

"""
This script runs the whole analysis for the full brain model 

It assumes the RDM and MDS are already computed (since they only take the visual system voxels into account)

I would suggest to not run this script, but each part individually. I'm writing this mainly to give a pipeline of what needs
to run. Although, if large amount of RAM (around 200G) is available, it should be fine 

The fitting is parralelized, if many CPUs are available I would suggest changing the n_jobs variable. 

On our machine with 6 CPUs, it took close to a month and a half to run
"""

if __name__ == "__main_full__":
    # get the betas
    load_betas_full(subj_list, sessions)

    # clean them; necessary due to missing values 
    clean_betas(subj_list)

    # Do the fitting. TAKES AGES
    gaussian_fit_full(subj_list, rois, params, mode=mode, n_jobs=6)

    # Computes the variance after fitting to avoid crashes
    variance_full_brain(subj_list, rois)

    # No model creation is needed: we just want to see what happens.
    # As before, the exporting to the cortical surface happens in a dedicated notebook

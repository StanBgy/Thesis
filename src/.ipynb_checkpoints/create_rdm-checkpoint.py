import os
import time
import numpy as np
from utils.utils import *
from scipy.spatial.distance import pdist 
from nsddatapaper_rsa.utils.utils import mds

"""
This file takes the betas, the mask and computes
the RDM based on the pdist scipy function 

The mask were created using a matlab script, available in the repo 
(credit to Luis for that)

The RDMs are then stored under the projects directory, and can be access later on 

Then, we used these RDM (or ones previously computed) to get the corresponding MDS 

"""


def create_rdm(list_subj, mode='averaged'):

    targetspace = "nativesurface"

    for i, sub in enumerate(list_subj):
        start_sub = time.time()
        print(f'Enter {sub} at {time.strftime("%H:%M:%S", time.gmtime(start_sub))}')

        maskdata_file = os.path.join(mask_dir, sub, f'short.{sub}.testrois.npy')
        maskdata = np.load(maskdata_file, allow_pickle=True).astype(int)
        
        betas_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_{mode}.npy') 
        betas = np.load(betas_file, allow_pickle=True).astype(np.float32)

        for mask_name in rois.keys():
            start_mask_name = time.time() 
            print(
            f'\tEnter ROI {mask_name} at: {time.strftime("%H:%M:%S", time.gmtime(start_mask_name))}'
            )
            rdm_file = os.path.join(rdm_dir, sub, f'{sub}_{mask_name}_fullrdm_correlation_{mode}.npy')

            mds_file = os.path.join(mds_dir, sub, f'{sub}_{mask_name}_mds_{mode}.npy')

            if not os.path.exists(rdm_file):
                roi = [rois[mask_name]]  # simpler that way, but does not allow for extended rois list. should not be an issue 
                vs_mask = np.array([e in roi for e in maskdata])
                masked_betas = betas[vs_mask, :]
                good_vox = [np.sum(np.isnan(x)) == 0 for x in masked_betas]

                if np.sum(good_vox) != len(good_vox):
                    print(f'\t\t\tfound some NaN for ROI: {mask_name} - {sub}')
    
                masked_betas = masked_betas[good_vox, :]

                # Transpose needed for correlation distance 
                X = masked_betas.T

                print(f'\t\t\tcomputing {mode} RDM for roi: {mask_name}')
                start_time = time.time()
                rdm = pdist(X, metric='correlation')

                if np.any(np.isnan(rdm)):
                    raise ValueError

                elapsed_time = time.time() - start_time
                print(
                    'elapsedtime: ',
                    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
                )

                print(f'\t\t\tsaving full rdm for {mask_name} : {sub}')
                np.save(
                    rdm_file,
                    rdm
                )

            if not os.path.exists(mds_file):

                rdm = np.load(rdm_file, allow_pickle=True).astype(np.float32)
                print(f'\t\t\tComputing {mode} MDS for {mask_name}')

                start_mds = time.time()

                # compute mds
                mds_out = mds(rdm).astype(np.float32)

                elapsed_time = time.time() - start_mds

                print(f'Time elapsed when computing MDS for {sub} ',
                      f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
                )

                print(f'\t\t\tsaving MDS for {mask_name} : {sub}')
                np.save(mds_file, mds_out)

            print(
            f'\t\tTime elapsed during {mask_name}: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start_mask_name))}'
            )

        print(
        f'\t\tTime elapsed during {sub}: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start_sub))}'
        )



# create_rdm(subjects_sessions)

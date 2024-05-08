import os
import time
import numpy as np
from utils.utils import *
from scipy.spatial.distance import pdist 
from nsddatapaper_rsa.utils.utils import mds
import nibabel as nib

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

      #  if np.isnan(betas).any():# drop the full row if there's some nan (so drop the whole voxel)  -> annoying but necessary, otherwise RDM breaks
       #     betas = betas[~np.isnan(betas).any(axis=1), :]    # This is fine, works as intended 

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
                print(masked_betas.shape)

                if np.isnan(masked_betas).any():# drop the full row if there's some nan (so drop the whole voxel)  -> annoying but necessary, otherwise RDM breaks
                    masked_betas = masked_betas[~np.isnan(masked_betas).any(axis=1), :]
                    
                    
                        # This is fine, works as intended. But this should be done before. 
                        

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

        
        if np.isnan(betas).any():
              # SUBJ06 AND SUBJ08 Have NaNs for some fucking reason: need to remove and replace. Can't do before because of the masking
            # which means I need to change the masks
            # I know for the hemishpere mask that they have been stack horizontally (its in luis' code) 
            # so if a voxel is below the cutoff, its in LH. if its above, its in RH
            if mode == 'train':
                betas_test_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_test.npy')  # also do it for test 
                betas_test = np.load(betas_test_file, allow_pickle=True).astype(np.float32)
                betas_test = betas_test[~np.isnan(betas).any(axis=1)]
                print(betas_test.shape)
                np.save(betas_test_file, betas_test)
                betas_mask_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy') # also do it for mask
                print('FIXING MASKS')
                betas_mask = np.load(betas_mask_file, allow_pickle=True)
                print(betas_mask.shape)
                betas_mask = betas_mask[~np.isnan(betas).any(axis=1)]
                betas_mask_file = os.path.join(mask_dir, sub, f'short.reduced.nans.{sub}.testrois.npy')
                np.save(betas_mask_file, betas_mask)
            maskdata_reduced_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy')
            maskdata_reduced = np.load(maskdata_reduced_file).astype(int)
            maskdata_reduced = maskdata_reduced[~np.isnan(betas).any(axis=1)]
            maskdata_reduced_file_new = os.path.join(mask_dir, sub, f'short.reduced.nans.{sub}.testrois.npy')  # also rewrite the mask (needed for model building)
            np.save(maskdata_reduced_file_new, maskdata_reduced)


            maskdata_lh_path = os.path.join(mask_dir, sub , f'lh.{sub}.testrois.mgz')
            maskdata_lh = nib.load(maskdata_lh_path).get_fdata().squeeze()
            maskdata_rh_path = os.path.join(mask_dir, sub , f'rh.{sub}.testrois.mgz')
            maskdata_rh = nib.load(maskdata_rh_path).get_fdata().squeeze()

            lh_indices = np.where((maskdata_lh >= 1) & (maskdata_lh <= 15))[0] 
            rh_indices = np.where((maskdata_rh >= 1) & (maskdata_rh <= 15))[0] 
            # I am storing the indexes of the mask corresponding to our ROIs so I can delete said indexes, without reformatting the whole mask 

            indices_to_detele_lh = lh_indices[np.isnan(betas[:lh_indices.shape[0]]).any(axis=1)] 
            indices_to_detele_rh = rh_indices[np.isnan(betas[lh_indices.shape[0]:]).any(axis=1)] 
            # lh_indices length is also the number of voxels in LH, so we know which voxel is in which hemi

         #   maskdata_lh_new = np.delete(maskdata_lh, indices_to_detele_lh)  delete messes it up later on when printing on the surface 
          #  maskdata_rh_new = np.delete(maskdata_rh, indices_to_detele_rh)

            maskdata_lh[indices_to_detele_lh] = 0
            maskdata_rh[indices_to_detele_rh] = 0

            maskdata_lh_img = nib.Nifti2Image(maskdata_lh, affine=None)
            maskdata_rh_img = nib.Nifti2Image(maskdata_rh, affine=None)

            lh_path = os.path.join(mask_dir, sub, f'lh.{sub}.nans.testrois.mgz')
            rh_path = os.path.join(mask_dir, sub, f'rh.{sub}.nans.testrois.mgz')
            nib.save(maskdata_lh_img, lh_path)
            nib.save(maskdata_rh_img, rh_path) 

            
            betas = betas[~np.isnan(betas).any(axis=1), :] # For the future: might be better to put these values as 0 (masking them, in a way) 
            
            np.save(betas_file, betas)

          #  betas = np.nan_to_num(betas, copy=False) #default replacement is 0.0
          #  np.save(betas_file, betas)
            

        print(
        f'\t\tTime elapsed during {sub}: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start_sub))}'
        )



# create_rdm(subjects_sessions)

import os
import nsd_access 
import numpy as np 
from utils.utils import * 
import nibabel as nib

def clean_betas(subj_list):
    """
    The betas gathered from the whole brain have some NaNs. 
    This code deals with that and modifiy the neuroimaging mask as well 
    """
    for sub in subj_list:
        counter = 0 

        maskdata_lh_path = os.path.join(mask_dir, sub , f'lh.{sub}.testrois.mgz')
        maskdata_lh = nib.load(maskdata_lh_path).get_fdata().squeeze()
        maskdata_rh_path = os.path.join(mask_dir, sub , f'rh.{sub}.testrois.mgz')
        maskdata_rh = nib.load(maskdata_rh_path).get_fdata().squeeze()

        maskdata = np.hstack([maskdata_lh, maskdata_rh])
        print(maskdata.shape)
        new_mask = []
        
        for i in range(5): 
            betas_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_nativesurface_train_full_{i}.npy')
            betas = np.load(betas_file, mmap_mode = 'r') 
            print(betas.shape)
            betas_test_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_nativesurface_test_full_{i}.npy')
            betas_test = np.load(betas_test_file, mmap_mode = 'r')

          
            if np.isnan(betas).any():
                
                betas_test = betas_test[~np.isnan(betas).any(axis=1)]
                
           
           
                n = np.delete(maskdata[counter:counter + betas.shape[0]], np.where(np.isnan(betas).any(axis=1)))
                new_mask.append(n)
                new_betas = betas[~np.isnan(betas).any(axis=1)]
            
                print(n.shape)

                assert new_betas.shape[0] == betas_test.shape[0]
                assert new_betas.shape[0] == n.shape[0]
                
                np.save(betas_file, new_betas)
                np.save(betas_test_file, betas_test)

            else: 
                new_mask.append(maskdata[counter:counter + betas.shape[0]])
            counter += betas.shape[0]

        maskdata= np.concatenate(new_mask)
        maskdata_lh_new = maskdata[:maskdata_lh.shape[0]]
        maskdata_rh_new = maskdata[maskdata_lh.shape[0]:]

        maskdata_lh_img = nib.Nifti2Image(maskdata_lh_new, affine=None)
        maskdata_rh_img = nib.Nifti2Image(maskdata_rh_new, affine=None)

        lh_path = os.path.join(mask_dir, sub, f'lh.{sub}.nans_full.testrois.mgz')
        rh_path = os.path.join(mask_dir, sub, f'rh.{sub}.nans_full.testrois.mgz')
        nib.save(maskdata_lh_img, lh_path)
        nib.save(maskdata_rh_img, rh_path) 

            


clean_betas(subj_list)
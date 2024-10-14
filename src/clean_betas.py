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
        counter, counter_del = 0 , 0 

        maskdata_lh_path = os.path.join(mask_dir, sub , f'lh.{sub}.testrois.mgz')
        maskdata_lh = nib.load(maskdata_lh_path).get_fdata().squeeze()
        maskdata_rh_path = os.path.join(mask_dir, sub , f'rh.{sub}.testrois.mgz')
        maskdata_rh = nib.load(maskdata_rh_path).get_fdata().squeeze()

        maskdata = np.hstack([maskdata_lh, maskdata_rh])
        print(maskdata.shape)
        new_mask, modified_mask = [], []
        cutoff = maskdata_lh.shape[0]  # cutoff takes into account the number of deleted voxels as well 
       # voxels_deleted = 0
        
        for i in range(5): 
            betas_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_nativesurface_train_full_{i}.npy')
            betas_file_new = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_nativesurface_train_full_{i}_fix.npy')
            betas = np.load(betas_file, mmap_mode = 'r') 
            print(betas.shape)
            betas_test_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_nativesurface_test_full_{i}.npy')
            betas_test_file_new = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_nativesurface_test_full_{i}_fix.npy')
            betas_test = np.load(betas_test_file, mmap_mode = 'r')

          
            if np.isnan(betas).any():
                
                betas_test = betas_test[~np.isnan(betas).any(axis=1)]

                if counter < cutoff:
                    voxels_deleted = np.where(np.isnan(betas).any(axis=1))[0].shape[0]
                    cutoff -= voxels_deleted
                elif counter >= cutoff: # this is when we get to the right hemi 
                    voxels_deleted = np.where(np.isnan(betas[cutoff-counter:]).any(axis=1))[0].shape[0]
                    cutoff -= voxels_deleted 
                    
                
           
                n = np.delete(maskdata[counter_del:counter_del + betas.shape[0]], np.where(np.isnan(betas).any(axis=1))) # this has to be coherent with the deletion
                new_mask.append(n)
                new_betas = betas[~np.isnan(betas).any(axis=1)]
                print(f'DELETED SHAPE :')
                print(n.shape)
                mask_to_modify = maskdata[counter:counter + betas.shape[0]]
               
                mask_to_modify[np.where(np.isnan(betas).any(axis=1))] = -100 
          #      mask_to_modify[0] = -100
               # arbitrary low value to be able to filter it out later 
                modified_mask.append(mask_to_modify)

                print('MODIFIED SHAPE')
                print(mask_to_modify.shape)

                assert new_betas.shape[0] == betas_test.shape[0]
                assert new_betas.shape[0] == n.shape[0]
                
                np.save(betas_file_new, new_betas)
                np.save(betas_test_file_new, betas_test)
                counter += betas.shape[0]
                counter_del += new_betas.shape[0]

            else: 
                new_mask.append(maskdata[counter:counter + betas.shape[0]])
                modified_mask.append(maskdata[counter:counter + betas.shape[0]])
                counter += betas.shape[0]
                counter_del += betas.shape[0]
            

        maskdata_del= np.concatenate(new_mask)
        maskdata_lh_new = maskdata_del[:cutoff]
        maskdata_rh_new = maskdata_del[cutoff:]

        mask_mod = np.concatenate(modified_mask)
        maskdata_lh_full_new = mask_mod[:maskdata_lh.shape[0]]
        maskdata_rh_full_new = mask_mod[maskdata_lh.shape[0]:]

        maskdata_lh_img = nib.Nifti2Image(maskdata_lh_new, affine=None)
        maskdata_rh_img = nib.Nifti2Image(maskdata_rh_new, affine=None)

        maskdata_lh_full_img = nib.Nifti2Image(maskdata_lh_full_new, affine=None)
        maskdata_rh_full_img = nib.Nifti2Image(maskdata_rh_full_new, affine=None)

        lh_path = os.path.join(mask_dir, sub, f'lh.{sub}.nans_full_del.testrois.mgz')
        rh_path = os.path.join(mask_dir, sub, f'rh.{sub}.nans_full_del.testrois.mgz')  # del is the one with deleted values

        lh_full_path = os.path.join(mask_dir, sub, f'lh.{sub}.nans_full.testrois.mgz')
        rh_full_path = os.path.join(mask_dir, sub, f'rh.{sub}.nans_full.testrois.mgz')
        print(f'mask shape : {maskdata.shape}')
        print(f'saving mask for {sub}')
        
        
        nib.save(maskdata_lh_img, lh_path)
        nib.save(maskdata_rh_img, rh_path) 

        nib.save(maskdata_lh_full_img, lh_full_path)
        nib.save(maskdata_rh_full_img, rh_full_path)

            


#clean_betas(subj_list)

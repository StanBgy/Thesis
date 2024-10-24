import os
import nsd_access 
import numpy as np 
from utils.utils import betas_dir, nsd_dir, proj_dir, mask_dir, sessions, subj_list, data_dir
from utils.split_condition import split_conditions
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas 
from nsddatapaper_rsa.utils.utils import average_over_conditions
from nsd_access import NSDAccess
import nibabel as nib


"""
Fetch  betas from all participants, and average them based on conditions
The conditions needs to be stored in the nsd_dir, then under ppdata
The responses.tsv files for each participants can be found in the original dataset, under 
their amazon webservice database 

I will try to provide a linux command to download it (and everything else) 

The betas are then stored in the appropriately named directory, and will 
be used for the RDM and fitting later on. 

I also allow for train-test spliiting
Default is None for simplicity sake

For loading the whole brain, refer to the load_betas_full.py; splitting the data was needed to not overload the RAM 
"""



targetspace = 'nativesurface'



def load_betas(subs, sessions, targetspace, mode='averaged'):
    for i, sub in enumerate(subs):
        maskdata_file = os.path.join(mask_dir, sub, f'{sub}.testrois.npy')
        maskdata_long = np.load(maskdata_file,allow_pickle=True)
        maskdata_long_bool = (maskdata_long > 0)
        maskdata_long_bool
        conditions = get_conditions(nsd_dir, sub, sessions[i])

        conditions = np.asarray(conditions).ravel()
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions]

        conditions_sampled = conditions[conditions_bool]

    # find the subject's unique condition list (sample pool)
        sample = np.unique(conditions[conditions_bool])

        conditions_to_save = [
            True if np.sum(conditions == x) >= 1 else False for x in conditions]
        saved_conditions = np.unique(conditions[conditions_to_save])
        conditions_list_path = os.path.join(data_dir, 'conditions', sub, f'{sub}.conditions.npy')
        np.save(conditions_list_path, saved_conditions, allow_pickle=True)

        if mode == 'averaged': 

            betas_mean_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_{mode}.npy') 

            if not os.path.exists(betas_mean_file):

                betas_mean = get_betas(
                        nsd_dir, 
                        sub,
                        sessions[i],
                        mask=maskdata_long_bool,
                        targetspace=targetspace,
                )
                print(f'concatenating betas for {sub}')
                betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

                print(f'Now averaging them')
                betas_mean = average_over_conditions(
                    betas_mean,
                    conditions,
                    conditions_sampled
                ).astype(np.float32)


                print(f'Saving conditions averaged betas')
                np.save(betas_mean_file, betas_mean)

            else:
                print(f'loading betas for {sub}')
                betas_mean = np.load(betas_mean_file, allow_pickle=True)
        

        if mode == "train":
            betas_train_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_train.npy')
            betas_test_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_test.npy')
            betas_mask_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_train_test_mask.npy')

            if not os.path.exists(betas_train_file) or not os.path.exists(betas_test_file) or not os.path.exists(betas_mask_file):
                print(f'\t\tcreating training and test split of betas for {sub}')

                betas_mean = get_betas(
                            nsd_dir,
                            sub,
                            sessions[i],
                            mask=maskdata_long_bool,
                            targetspace=targetspace,
                        )

                print(f'\t\t concatenating betas for {sub}')

                betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

                ### Do the splitting
                betas_train, betas_test, train_test_mask, train_test_conditions = split_conditions(betas_mean, conditions, conditions_sampled)


                print(f'\t\tsaving training betas for {sub}')
                np.save(betas_train_file, betas_train, allow_pickle=False)

                print(f'\t\tsaving testing betas for {sub}')
                np.save(betas_test_file, betas_test)

                print(f'\t\tsaving training-testing mask for {sub}')
                np.save(betas_mask_file, train_test_mask)

                
            else:
                print(f'\t\tfiles exist for {sub}')


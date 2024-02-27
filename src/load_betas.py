import os
import nsd_access 
import numpy as np 
from utils.utils import betas_dir, nsd_dir, proj_dir, sessions
from utils.split_condition import split_conditions
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas 
from nsd_access import NSDAccess
import nibabel as nib


"""
Fetch  betas from all participants, and average them based on conditions
The condiitons needs to be stored in the nsd_dir, then under ppdata
The responses.tsv files for each participants can be found in the original dataset, under 
their amazon webservice database 

I will try to provide a linux command to download it (and everything else) 

The betas are then stored in the appropriately named directory, and will 
be used for the RDM and fitting later on. 

I also allow for train-test spliiting, based on Luis' function
Default is None for simplicity sake
"""

n_jobs = 2 # Local machine, don't go too crazy
n_subjects = 8 # only try sub 1 here

nsda = NSDAccess(nsd_dir)


targetspace = 'nativesurface'


# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

def load_betas(subs, sessions, targetspace, split=None):
    for i, sub in enumerate(subs):
        conditions = get_conditions(nsd_dir, sub, sessions[i])

        conditions = np.asarray(conditions).ravel()
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions]

        conditions_sampled = conditions[conditions_bool]

    # find the subject's unique condition list (sample pool)
        sample = np.unique(conditions[conditions_bool])

        if not split: 

            betas_mean_file = os.path.join(betas_dir, f'{subs[0]}_betas_list_{targetspace}_averaged.npy') 

            if not os.path.exists(betas_mean_file):

                betas_mean = get_betas(
                    nsd_dir, 
                    sub,
                    sessions[i],
                    targetspace=targetspace,
                )
                print(f'concatenating betas for {subs[0]}')
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
        

        if split:
            # not averaged over conditions? 
            betas_train_file = os.path.join(betas_dir, f'{subs[0]}_betas_list_{targetspace}_train.npy')
            betas_test_file = os.path.join(betas_dir, f'{subs[0]}_betas_list_{targetspace}_test.npy')
            betas_mask_file = os.path.join(betas_dir, f'{subs[0]}_betas_list_{targetspace}_train_test_mask.npy')

            if not os.path.exists(betas_train_file) or not os.path.exists(betas_test_file) or not os.path.exists(betas_mask_file):
                print(f'\t\tcreating training and test split of betas for {sub}')

                betas_mean = get_betas(
                        nsd_dir,
                        sub,
                        sessions[i],
                        targetspace=targetspace,
                        )

                print(f'\t\t concatenating betas for {sub}')

                betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

                betas_train, betas_test, train_test_mask, train_test_conditions = split_conditions(betas_mean, conditions_betas, conditions_sampled)

                print(f'\t\tsaving training betas for {sub}')
                np.save(betas_train_file, betas_train)

                print(f'\t\tsaving testing betas for {sub}')
                np.save(betas_test_file, betas_test)

                print(f'\t\tsaving training-testing mask for {sub}')
                np.save(train_test_mask_file, train_test_mask)
            else:
                print(f'\t\tfiles exist for {sub}')

# load_betas(subs, sessions, targetspace, split=True)

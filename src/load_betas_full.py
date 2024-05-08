import os
import nsd_access 
import numpy as np 
from utils.utils import betas_dir, nsd_dir, proj_dir, mask_dir, sessions, subj_list, data_dir
from utils.split_condition import split_conditions
from utils.get_betas_mod import get_betas_mod
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas 
from nsddatapaper_rsa.utils.utils import average_over_conditions
from nsd_access import NSDAccess
import nibabel as nib

def load_betas_full(subs, sessions, targetspace='nativesurface', mode='train'):
    """
    Attempt to get the betas for the whole brain data

    The mode is not implemented yet, since we only use train

    Otherwise this code will fetch the betas for each session separately, 
    and then save them in numpy to make life easier 

    Note: on our machine (34GB of RAM + 96GB of Swap) this crashes a few time (around one time for participant)
    There is probably a way to do this without overloading the RAM that would be worth exploring
    The biggest bottleneck is np.concatenate, so I would explore this if needs be 
    But otherwise, reloading the script is not too much of a hassle since everthing gets saved
    """
    for i, sub in enumerate(subs):
        conditions = get_conditions(nsd_dir, sub, sessions[i])
    
        conditions = np.asarray(conditions).ravel()
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions]
    
        conditions_sampled = conditions[conditions_bool]
    
        # find the subject's unique condition list (sample pool)
        sample = np.unique(conditions[conditions_bool])

        betas_train_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_{targetspace}_train_full_0.npy')  # store these on the HDD 
        betas_test_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_{targetspace}_test_full.npy')
        betas_mask_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_{targetspace}_train_test_mask_full.npy')

        betas = []
        if os.path.exists(betas_train_file):
            print(f'Betas exists for {sub}')
            continue
        for j in range(sessions[i]):
            j += 1
            session_betas_file = os.path.join(nsd_dir, 'full_brain', sub, f'betas_sessions_{j}.npy')
            if not os.path.exists(session_betas_file):
                session_betas = get_betas_mod(
                nsd_dir,
                sub, 
                j,
                targetspace=targetspace)

                print(f'Save betas for {sub} on session {j}')
                np.save(session_betas_file, session_betas)
                del session_betas
            else: 
                print(f"Sessions {j} exists for {sub}")


        for j in range(sessions[i]):
            j +=1
            betas.append(np.load(os.path.join(nsd_dir, 'full_brain', sub, f'betas_sessions_{j}.npy'), mmap_mode='r').squeeze())
            if sub == 'subj03' or sub == 'subj06':
                divider = 8
            else:
                divider = 10
            
            if j != 0 and j % divider == 0:
                betas_contact_file = os.path.join(nsd_dir, 'full_brain', sub, f'betas_concat_at_{j}.npy')
                if not os.path.exists(betas_contact_file):
                    betas = np.concatenate(betas, axis=1).astype(np.float32)
                
                    print(f'Saving concatenated betas for sessions {j}')
                    np.save(betas_contact_file, betas)
                    betas = []
                else:
                    print(f"Concat at {j} already exists for {sub}")
        del betas
        
        full_betas = os.path.join(nsd_dir, 'full_brain', sub, f'full_betas_{sub}.npy')
        if not os.path.exists(full_betas):
            print(f'CONCATENATING BETAS FOR {sub}')
            if sub == 'subj03' or sub == 'subj06':
                all_betas = [np.load(f, mmap_mode='r') for f in [f"{nsd_dir}full_brain/{sub}/betas_concat_at_8.npy", f"{nsd_dir}full_brain/{sub}/betas_concat_at_16.npy", \
                                                          f"{nsd_dir}full_brain/{sub}/betas_concat_at_24.npy", f"{nsd_dir}full_brain/{sub}/betas_concat_at_32.npy"]]#deal with these two special case, a bit stupid of a solution but does it 
                all_betas = np.concatenate(all_betas, axis=1)
            if sub == 'subj04' or sub == 'subj08':
                all_betas = [np.load(f, mmap_mode='r') for f in [f"{nsd_dir}full_brain/{sub}/betas_concat_at_10.npy", f"{nsd_dir}full_brain/{sub}/betas_concat_at_20.npy", \
                                                          f"{nsd_dir}full_brain/{sub}/betas_concat_at_30.npy"]]
                all_betas = np.concatenate(all_betas, axis=1)
            if sub != 'subj03' and sub!= 'subj04' and sub != 'subj06' and sub != 'subj08': # ????
                all_betas = [np.load(f, mmap_mode='r') for f in [f"{nsd_dir}full_brain/{sub}/betas_concat_at_10.npy", f"{nsd_dir}full_brain/{sub}/betas_concat_at_20.npy", \
                                                          f"{nsd_dir}full_brain/{sub}/betas_concat_at_30.npy", f"{nsd_dir}full_brain/{sub}/betas_concat_at_40.npy"]]
                all_betas = np.concatenate(all_betas, axis=1)
        #    all_betas = np.concatenate(all_betas, axis=1)
            np.save(full_betas, all_betas)
            del all_betas
        else:
            print(f"FULL concat already exists for {sub}")

       
        all_betas = np.load(full_betas, mmap_mode='r') #mmap_mode make a memory map on disk, so it doesnt overload the ram, and is quite fast
        split_betas = np.array_split(all_betas, 5, axis=0)# split in 5 equal (or close to) chunks 
        print(f'Splitting train_test and average over conditions for {sub}')
        del all_betas
    
        for i in range(5):

            betas_train_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_{targetspace}_train_full_{i}.npy')
            betas_test_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_{targetspace}_test_full_{i}.npy')
            betas_mask_file = os.path.join(nsd_dir, 'full_brain', sub, f'{sub}_betas_list_{targetspace}_train_test_mask_full_{i}.npy')
            if not os.path.exists(betas_train_file):
                betas_train, betas_test, train_test_mask, train_test_conditions = split_conditions(split_betas[i], conditions, conditions_sampled)

                print(f'\t\tsaving training betas for {sub}, chunk {i}')
                np.save(betas_train_file, betas_train, allow_pickle=False)

                print(f'\t\tsaving testing betas for {sub}, chunk {i}')
                np.save(betas_test_file, betas_test)

                print(f'\t\tsaving training-testing mask for {sub}, chunk {i}')
                np.save(betas_mask_file, train_test_mask)

        del split_betas
      
            
      
            

        

load_betas_full(subj_list, sessions)

    
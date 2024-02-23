import os
import nsd_access 
import numpy as np 
from utils.utils import betas_dir, nsd_dir, proj_dir
from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas 
from nsddatapaper_rsa.utils.utils import average_over_conditions
from nsd_access import NSDAccess
import nibabel as nib

n_jobs =12 
n_sessions = 40
n_subjects = 1 # only try sub 1 here

nsda = NSDAccess(nsd_dir)

outpath = os.path.join(betas_dir, 'roi_analyses')
if not os.path.exists(outpath):
    os.makedirs(outpath)

targetspace = 'fsaverage'

# I found this in the paper repo -> mainfigures -> SCIENCE.RSA
lh_file = os.path.join(proj_dir, 'lh.highlevelvisual.mgz')
rh_file = os.path.join(proj_dir, 'rh.highlevelvisual.mgz')


# load them
maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

maskdata = np.hstack((maskdata_lh, maskdata_rh))

ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}

roi_names = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']

# sessions
n_sessions = 40

# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

conditions = get_conditions(nsd_dir, subs[0], n_sessions)

conditions = np.asarray(conditions).ravel()
conditions_bool = [
    True if np.sum(conditions == x) == 3 else False for x in conditions]

conditions_sampled = conditions[conditions_bool]

# find the subject's unique condition list (sample pool)
sample = np.unique(conditions[conditions_bool])
assert sample.shape[0] == 10000

betas_mean_file = os.path.join(outpath, f'{subs[0]}_betas_list_{targetspace}_concat.npy') 
if not os.path.exists(betas_mean_file):
    betas_mean = get_betas(
        nsd_dir, 
        subs[0],
        n_sessions,
        targetspace=targetspace,
    )
    print(f'concatenating betas for {subs[0]}')
    betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

    print(f'saving concatenated betas')

    np.save(betas_mean_file, betas_mean)

    print(f'Now averaging them')
   # betas_mean = average_over_conditions(
    #    betas_mean,
     #   conditions,
    #    conditions_sampled
    #).astype(np.float32)

    #print(f'Saving conditions averaged betas')
    #np.save(betas_mean_file, betas_mean)

else:
    print(f'loading betas for {subs[0]}')
    betas_mean = np.load(betas_mean_file, allow_pickle=True)

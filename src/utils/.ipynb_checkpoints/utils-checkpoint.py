import os
import numpy as np


"""
Some needed variables for all scripts
"""

targetspace = 'nativesurface'

base_dir  = '/media/Working/stan-thesis/'

data_dir = os.path.join(base_dir, 'data')
mask_dir = os.path.join(data_dir, 'mask')
label_dir = os.path.join(data_dir, 'nsddata')

# Proj dir is where everything that we compute
proj_dir = os.path.join(base_dir, 'projects')
betas_dir = os.path.join(proj_dir, 'betas')
rdm_dir = os.path.join(proj_dir, 'rdm')
mds_dir = os.path.join(proj_dir, 'MDS')

fits_dir = os.path.join(proj_dir, 'fits')
param_dir = os.path.join(proj_dir, 'stats', 'parametric_test') 
models_dir = os.path.join(proj_dir, 'serialised_models')
results_dir = os.path.join(proj_dir, 'results')


#nsd_dir = os.path.join(data_dir, 'NSD') # SSD. Sometimes only works in there. Don't ask
nsd_dir = os.path.join('/media/harveylab/STORAGE1_NA/NSD/') # Now on HDD. Works only during odd months, full moon, and if Jupiter is at a 45 degrees angle with Earth

nsd_dir = os.path.join(data_dir, 'NSD')


#sessions = [37, 37, 29, 27, 37, 29, 37, 27]
sessions = [40, 40, 32, 30, 40, 32, 40, 30]
subjects_sessions = {i: (f'subj0{i}',sessions[i-1]) for i in range(1, 9)}

subj_list = [sub[0] for sub in list(subjects_sessions.values())]
rois = {'V1': 1, 'V2': 2, 'V3': 3, 'hV4': 4, 'VO-1': 5, 'VO-2': 6,
 'PHC-1': 7, 'PHC-2': 8, 'LO-1': 9, 'LO-2': 10, 'TO-1': 11, 'TO-2': 12
}

rois_long = {'V1v': 1, 'V1d': 2, 'V2v': 3, 'V2d': 4, 'V3v': 5, 'V3d': 6, 'hV4': 7, 'VO-1': 8,
 'VO-2': 9, 'PHC-1': 10, 'PHC-2': 11, 'LO-1': 12, 'LO-2': 13, 'TO-1': 14, 'TO-2': 15
}

reduced = {'V1': ['V1v', 'V1d'], 'V2': ['V2v', 'V2d'], 'V3': ['V3v', 'V3d']}
# Reversed: 
sior = {v:k for k, v in rois.items()}

params = {'random': True, 
          'initial': (np.array([-0.4, -0.4, 0.01, 0.1, - 2]), np.array([0.4, 0.4, 2, 10, 2])), 
           'bounds': (np.array([-1.05, -1.05, 0.01, 1e-08, - np.inf]),np.array([1.05, 1.05, np.inf, np.inf, np.inf])),
           'loss': 'linear',
           'method': 'trf'
        }

models = ['wself', 'oself']

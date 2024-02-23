import os 

base_dir = '/media/harveylab/STORAGE1_NA/stan-thesis/'
mask_dir = '/media/harveylab/STORAGE1_NA/luis-thesis/spatial_neural_responses/data/nsddata/freesurfer'
print(os.path.exists(base_dir))

data_dir = os.path.join(base_dir, 'data')

mds_dir = os.path.join(data_dir, 'MDS')
proj_dir = os.path.join(base_dir, 'projects', 'NSD')
betas_dir = os.path.join(proj_dir, 'rsa')
sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
betas_dir = os.path.join(proj_dir, 'rsa')
models_dir = os.path.join(betas_dir, 'serialised_models')


nsd_dir = os.path.join(data_dir, 'NSD')

targetspace = 'nativesurface'  # ?????
# sessions = [37, 37, 29, 27, 37, 29, 37, 27]
sessions = [40, 40, 32, 30, 40, 32, 40, 30]
subjects_sessions = {i: (f'subj0{i}',sessions[i-1]) for i in range(1,9)}
rois = {'V1': 1, 'V2': 2, 'V3': 3, 'hV4': 4, 'VO-1': 5, 'VO-2': 6,
 'PHC-1': 7, 'PHC-2': 8, 'LO-1': 9, 'LO-2': 10, 'TO-1': 11, 'TO-2': 12
}

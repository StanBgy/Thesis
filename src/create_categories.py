import os 
import numpy as np 
from nsd_access import NSDAccess
from utils.utils import *

subject = 1
sub, n_sessions = subjects_sessions[subject]

nsda = NSDAccess(data_dir)

categories_betas_file = os.path.join(data_dir, 'categories_people_files', f'{sub}_{n_sessions}_categories_betas.npy')

conditions_betas = np.load('/home/stan/thesis-repo/data/conditions_betas/subj01_37_conditions_betas.npy', allow_pickle=True)

if not os.path.exists(categories_betas_file):
    categories_betas = nsda.read_image_coco_category(conditions_betas)
    print(f'\t\tsaving betas categories for {sub}')
    np.save(categories_betas_file, categories_betas)


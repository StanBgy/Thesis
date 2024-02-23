import os
import numpy as np
from nsddatapaper_rsa.utils.utils import mds

RDM = np.load('subj02_37_V1_fullrdm_correlation.npy', allow_pickle=False)
print(RDM.shape)




mds_V1 = mds(RDM, n_jobs=12)
print('saving mds')
np.save('MDS_V1_subj02.npy', MDS_V1)

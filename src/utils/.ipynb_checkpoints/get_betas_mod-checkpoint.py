import numpy as np
import nibabel as nb
from scipy.stats import zscore
from nsd_access import NSDAccess
import os
import h5py as h5


def get_betas_mod(nsd_dir, sub, ses_i, mask=None, targetspace='func1pt8mm'):
    """ betas = get_betas(nsd_dir, sub, n_sessions, mask, targatspace)

    Arguments:
    ___________

        nsd_dir (os.path): absolute path to the NSD data folder.

        sub (string): subject identifier (e.g. subj01)

        n_sessions (int): the number of sessions to fetch data from

        mask (bool or index, optional): logical mask (e.g. a specific roi)

        targetspace (str, optional): Data preparation space.
            Defaults to 'func1pt8mm'.

    Returns:
    __________

        array: numpy array of betas with shape features x conditioons

        @ StanBgy: modified file to only load one sessions
    """

    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    data_folder = os.path.join(
        nsda.nsddata_betas_folder,
        sub,
        targetspace,
        'betas_fithrf_GLMdenoise_RR')

    betas = []
    # loop over sessions
    # trial_index=0
    
    
    si_str = str(ses_i).zfill(2)

        # sess_slice = slice(trial_index, trial_index+750)
    print(f'\t\tsub: {sub} fetching betas for trials in session: {ses_i}')

        # we only want to keep the shared_1000
    this_ses = nsda.read_behavior(subject=sub, session_index=ses_i)

        # these are the 73K ids.
    ses_conditions = np.asarray(this_ses['73KID'])

    valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
    if valid_trials:

        if targetspace == 'fsaverage':
            conaxis = 1

            # load lh
            """
            @StanBgy change#: change file extenion to mgh
            and get_data to get_fdata (deprecated"
            """
            img_lh = nb.load(
                        os.path.join(
                            data_folder,
                            f'lh.betas_session{si_str}.mgh'
                            )
                    ).get_fdata().squeeze()

                # load rh
            img_rh = nb.load(
                        os.path.join(
                            data_folder,
                            f'rh.betas_session{si_str}.mgh'
                            )
                    ).get_fdata().squeeze()

                # concatenate
            all_verts = np.vstack((img_lh, img_rh))

                # mask
            if mask is not None:
                tmp = zscore(all_verts, axis=conaxis).astype(np.float32)

                    # you may want to get several ROIs from a list of ROIs at
                    # once
                if type(mask) == list:
                    masked_betas = []
                    for mask_is in mask:
                        tmp2 = tmp[mask_is, :]
                            # check for nans
                            # good = np.any(np.isfinite(tmp2), axis=1)
                        masked_betas.append(tmp2)
                else:
                    tmp2 = tmp[mask_is, :]
                    masked_betas = tmp2

                betas.append(masked_betas)
            else:
                betas.append(
                    (zscore(
                        all_verts,
                        axis=conaxis)).astype(np.float32)
                    )

        if targetspace == 'nativesurface':
            conaxis = 1

                # load lh
            """
            @StanBgy change#: change file extenion to mgh
            and get_data to get_fdata (deprecated"
            """
            img_lh = h5.File(
                        os.path.join(
                            data_folder,
                            f'lh.betas_session{si_str}.hdf5'
                            )
                    )

            img_lh = np.transpose(img_lh['betas'])

                # load rh
            img_rh = h5.File(
                        os.path.join(
                            data_folder,
                            f'rh.betas_session{si_str}.hdf5'
                            )
                    )

            img_rh = np.transpose(img_rh['betas'])

                # concatenate
            all_verts = np.vstack((img_lh, img_rh))

                # mask
            if mask is not None:
                tmp = zscore(all_verts, axis=conaxis).astype(np.float32)

                    # you may want to get several ROIs from a list of ROIs at
                    # once
                if type(mask) == list:
                    masked_betas = []
                    for mask_is in mask:
                        tmp2 = tmp[mask_is, :]
                            # check for nans
                            # good = np.any(np.isfinite(tmp2), axis=1)
                        masked_betas.append(tmp2)
                else:
                    tmp2 = tmp[mask, :]  # mistake in original repo, because why not
                    masked_betas = tmp2

                betas.append(masked_betas)
            else:
                betas.append(
                        (zscore(
                            all_verts,
                            axis=conaxis)).astype(np.float32)
                    )
                    
        else:
            conaxis = 1
            img = nb.load(
                os.path.join(data_folder, f'betas_session{si_str}.nii.gz'))

            if mask is not None:
                betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)[mask, :]*300).astype(np.int16)
                    )
            else:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)*300).astype(np.int16)
                    )

    return betas
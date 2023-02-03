
import numpy as np
import logging

from src.data import path_handling as P, file_handling as FH, toml_handling as TH
from src import constants as C
from src.rendering import blender_control as BC
from src.utils import general_utils as GU, data_utils as DU, spectra_utils as SU

# TODO plot resampling result to targets directory


def resample(set_name):
    """Resamples leaf spectra to lower resolution as defined in /sample_target/sampling.toml."""

    ids = FH.list_target_ids(set_name)
    if len(ids) < 1:
        raise RuntimeWarning(f"No targets to resample.")

    sampling = TH.read_sampling(set_name=set_name)

    if len(sampling) < 1:
        logging.info(f"No resampling defined. Returning without doing anything.")
        return

    for _, sample_id in enumerate(ids):
        target = TH.read_target(set_name=set_name, sample_id=sample_id)
        wls, refls, trans = map(list, zip(*target)) # unpack target to lists
        spectra = np.array((refls,trans))

        try:
            resampled = SU.resample(wls, spectra, new_wl=sampling)
        except IndexError as e:
            raise IndexError(f"Index error occurred probably because of empty entry in resampling file.") from e

        refls_new = resampled[0,:]
        trans_new = resampled[1,:]
        resampled_target = np.zeros((3, len(sampling)))
        resampled_target[0] = sampling
        resampled_target[1] = refls_new
        resampled_target[2] = trans_new
        resampled_target = np.transpose(resampled_target)
        TH.write_target(set_name=set_name, data=resampled_target, sample_id=sample_id, resampled=True)

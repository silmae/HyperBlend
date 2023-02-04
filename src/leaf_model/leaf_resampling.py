
import numpy as np
import logging

from src.data import path_handling as P, file_handling as FH, toml_handling as TH
from src import constants as C
from src.rendering import blender_control as BC
from src.utils import general_utils as GU, data_utils as DU, spectra_utils as SU
from src import plotter


def resample(set_name):
    """Resamples leaf spectra to lower resolution as defined in /sample_target/sampling.toml."""

    ids = FH.list_target_ids(set_name)
    if len(ids) < 1:
        logging.info(f"No targets to resample. Returning without doing anything.")
        return

    sampling = TH.read_sampling(set_name=set_name)

    if len(sampling) < 1:
        logging.info(f"No resampling defined. Returning without doing anything.")
        return

    for _, sample_id in enumerate(ids):
        target = TH.read_target(set_name=set_name, sample_id=sample_id)
        wls, refls, trans = DU.unpack_target(target=target)
        spectra = np.array((refls,trans))

        try:
            resampled = SU.resample(wls, spectra, new_wl=sampling)
        except IndexError as e:
            raise IndexError(f"Index error occurred probably because of empty entry in resampling file.") from e

        resampled_target = DU.pack_target(wls=sampling, refls=resampled[0,:], trans=resampled[1,:])
        TH.write_target(set_name=set_name, data=resampled_target, sample_id=sample_id, resampled=True)

        plotter.plot_resampling(set_name=set_name)

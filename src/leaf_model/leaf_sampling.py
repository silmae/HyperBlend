
import numpy as np
import logging

from src.data import path_handling as P, file_handling as FH, toml_handling as TH
from src import constants as C
from src.rendering import blender_control as BC
from src.utils import general_utils as GU, data_utils as DU, spectra_utils as SU
from src import plotter


def sampling_empty(set_name: str) -> bool:
    sampling = TH.read_sampling(set_name=set_name)
    if len(sampling) < 1:
        return True
    return False


def check_sampling(set_name: str) -> bool:
    """Check that leaf material parameters are solved with current sampling.

    :param set_name:
    :return:
        Returns `True` if either, sampling information file does not contain resampling
        wavelengths or if the sample results and set results have the same wavelengths
        as the sampling information file. Otherwise return `False`.
    """

    sampling = TH.read_sampling(set_name=set_name)

    if len(sampling) < 1:
        logging.info(f"No resampling defined. Sampling is thus OK and no further checks will be run.")
        return True

    set_result = TH.read_set_result(set_name=set_name)
    set_wls = np.array(set_result[C.key_set_result_wls])

    if not np.allclose(sampling, set_wls):
        logging.warning(f"Sampling data does not match set result wavelengths. "
                        f"Re-solve leaf material parameters to fix.")
        return False

    ids = FH.list_finished_sample_ids(set_name=set_name)
    for sample_id in ids:
        sample_result = TH.read_sample_result(set_name=set_name, sample_id=sample_id)
        sample_wls = sample_result[C.key_sample_result_wls]
        if not np.allclose(sampling, sample_wls):
            logging.warning(f"Sampling data does not match leaf sample {sample_id} wavelengths. "
                         f"Re-solve leaf material parameters to fix.")
            return False

    logging.info(f"All sampling checks passed for leaf measurement set '{set_name}'.")
    return True


def resample(set_name: str):
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
        # Read target in original resolution
        target = TH.read_target(set_name=set_name, sample_id=sample_id, resampled=False)
        wls, refls, trans = DU.unpack_target(target=target)
        spectra = np.array((refls,trans))

        try:
            resampled = SU.resample(wls, spectra, new_wl=sampling)
        except IndexError as e:
            raise IndexError(f"Index error occurred probably because of empty entry in resampling file.") from e

        resampled_target = DU.pack_target(wls=sampling, refls=resampled[0,:], trans=resampled[1,:])
        TH.write_target(set_name=set_name, data=resampled_target, sample_id=sample_id, resampled=True)

    plotter.plot_resampling(set_name=set_name)

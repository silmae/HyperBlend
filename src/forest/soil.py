
import os
import numpy as np
import logging

from src.gsv import gsv
from src.data import path_handling as PH
from src.utils import spectra_utils as SU


def find_gsv_soil_path(soil_name):
    """Finds path to a soil file that includes given soil name.

    Returns the first occurrence if multiple files match the name.
    """

    p_soil_dir = PH.path_directory_soil_data()
    for file_name in os.listdir(p_soil_dir):
        if soil_name in file_name:
            return PH.join(p_soil_dir, file_name)

    raise FileNotFoundError(f"Could not find soil '{soil_name}' from '{p_soil_dir}'.")


def load_soil(soil_name: str, forest_id: str = None, sampling=None):
    """Find a soil csv from root/soil/ and place a resampled copy of it to given forest scene.

    :param soil_name:
        Soil name to be searched. Must be one that exists in root/soil/.
    :param forest_id:
        Forest scene id where the result should be writeen.
    :param sampling:
        Resampling data as a list of wavelengths.
    """

    p = find_gsv_soil_path(soil_name=soil_name)
    soil_data = np.loadtxt(p)
    wls = soil_data[:,0]
    refls = soil_data[:,1]

    if sampling is not None:
        resampled_refls = SU.resample(original_wl=wls, original_val=refls, new_wl=sampling)
        return sampling, resampled_refls

    return wls, refls

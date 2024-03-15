"""
This script is an access point to load light spectra to be used
in big scene simulation.
"""

import numpy as np
import os
import logging

from src import constants as C
from src.data import path_handling as PH
from src.data import light_file_handling as LFH
from src.utils import spectra_utils as SU


def load_light(file_name: str = None, scene_id=None, sampling=None, lighting_type='sun'):
    """ Loads a lighting file and returns wavelengths and corresponding irradiances.

    Files formatted so that comment lines are prefixed with '#' and rest of the lines
    contain wavelength-irradiance pairs that can be casted to floats, they can be read
    directly. For spectra generated with either NASA PSG or SSolar GOA, the files are
    fixed so that they can be read directly later.

    :param file_name:
        Optional. If given, a file with this name is searched from light_data directory. If
        also scene_id is given, the scene directory is searched first before extending the
        search to light_data directory. If not given, lighting type must be given (either
        'sun' or 'sky' so that default light files can be loaded.
    :param scene_id:
        If `scene_id` is given, the scene directory is searched first before extending the
        search to light_data directory.
    :param sampling:
        List of wavelengths as floats. If given, sun data is resampled to wavelengths specified
        in the list. If sampling is None (default), the data is returned as raw.
    :param lighting_type:
        Lighting type either 'sun' or 'sky', default is 'sun'. This is used to get the default
        sun or sky file if `file_name` was not given. If both are given, the `file_name` has precedence.
    :return:
        (wls, irradiances) tuple where wls is a list of wavelengths (bands) and irradiances are
        corresponding list of irradiances. The length of the lists vary depending on given sampling.
    :raises:
        ValueError if `file_name` was not provided and `type` is not either 'sun' or 'sky'.
    """

    logging.info("Loading light data.")
    if file_name is None:
        if lighting_type == 'sun':
            file_name = C.file_default_sun
        elif lighting_type == 'sky':
            file_name = C.file_default_sky
        else:
            raise ValueError(f"Lighting file name was not provided. For loading one of the default files, "
                             f"expected file type either 'sun' or 'sky', was '{lighting_type}'.")

    if not file_name.endswith('.txt'):
        file_name = file_name + '.txt'

    path = _find_lighting_file(file_name, scene_id)

    wls, irradiances = LFH.read_light_file(path)

    if sampling is not None:
        new_irradiances = SU.resample(original_wl=wls, original_val=irradiances, new_wl=sampling)
        wls = sampling
        irradiances = new_irradiances

    logging.info("Light data loaded.")

    return np.array(wls), np.array(irradiances)


def _find_lighting_file(file_name: str, forest_id: str = None) -> str:
    """Attempts to find a light file with given filename.

    :param file_name:
        A file with this name is searched from light_data directory. If
        also scene_id is given, the scene directory is searched first before
        extending the search to light_data directory.
    :param forest_id:
        Optional. If not given, forest scene directory is not searched.
    :return:
        Path to found file.
    :raises FileNotFoundError:
        If the file is not found.
    :raises ValueError:
        If the file could not be found from the scene directories and `lighting_type` was not
        either 'sun' or 'sky'.
    """

    if forest_id is not None:
        logging.info(f"Trying to find lighting data from forest scene directory '{PH.path_directory_forest_scene(forest_id)}'.")
        p = PH.join(PH.path_directory_forest_scene(forest_id), file_name)
        if os.path.exists(p):
            logging.info(f"Light data found.")
            return p
        else:
            logging.info(f"Could not find sun data from scene directory. Now searching default directory.")

    p_dir = PH.path_directory_light_data()

    p = PH.join(p_dir, file_name)
    if os.path.exists(p):
        return p

    raise FileNotFoundError(f"File from '{p}' can not been found.")


if __name__ == '__main__':

    """
    Main for testing and debugging. 
    """

    import sys
    logging.basicConfig(stream=sys.stdout, level='DEBUG')

    # Test splitting GOA-generated file into sun and sky files
    # This should produce an error after the split is done
    try:
        wls, irradiances = load_light(file_name='goa_output')
    except RuntimeError:
        print("Error produced as is proper.")

    # Test loading the new files
    wls, irradiances = load_light(file_name='goa_output_sun')
    wls, irradiances = load_light(file_name='goa_output_sky')

    # Test loading the default sun and sky files
    wls, irradiances = load_light(lighting_type='sun')
    wls, irradiances = load_light(lighting_type='sky')

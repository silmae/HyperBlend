"""
Sun-related stuff.

Simple use-case:
    1. call load_sun with no filename and get the default sun irradiance integrated
        to whichever bandwith given.
Custom sun use-case:
    1. use https://psg.gsfc.nasa.gov/ to generate desired sun spectra
    2. download the spectra and save it to either sun_data directory (to be available
        for all scenes, or to top level directory of a certain scene.
    3. call load_sun with filename you saved the file in previous step. Integrated
        to whichever bandwith given.

How to generate spectra with NASA's Planetary Spectrum generator:
    1. Select template "Sun from Earth" and click "Load Spectra".
    2. If you want to select certain location and date on Earth, click
        "Change Object" button. Remember to click save settings!
    3. Click "Change Instrument"
        1. Set "Spectral range" to 400 - 2500 nm. Remember to change units
            from um to nm!
        2. Set "Resolution" to 1 nm and units from "Resolving power" to nm.
        3. Set "Spectrum intensity unit" to W/m2/um (spectral irradiance)
    4. Change other settings as you see fit, but know what you are doing!
    5. Click "Generate Spectra"
    6. From the first image, click "Download Spectra"
"""

import numpy as np
import os
import logging

from src import constants as C
from src.data import path_handling as PH
from src.data.nasa_psg_handler import read_psg_file
from src.utils import spectra_utils as SU


def load_light(file_name: str = None, forest_id=None, sampling=None, lighting_type='sun'):
    """ Loads a lighting file and returns wavelengths and corresponding irradiances.

    If given file is not compatible with HyperBlend, an attempt is made to fix it.
    Given file must contain wavelengths from 400 to 2500 (both inclusive) and have spectral
    resolution of 1 nm.

    :param file_name:
        Optional. If given, this file is searched from sun_data directory. If
        also scene_id is given, the search is extended to scene directory. Precedence
        is then for the file in the scene directory. If not given, the default sun file
        from the repository is used.
    :param forest_id:
        If forest_id is given, the search is extended to scene directory so that the scene
        directory is searched first and if found, it is returned. If not found, the sun_data
        directory is then searched.
    :param sampling:
        List of floats. If given, sun data is resampled to wavelengths specified in the list.
        If sampling is None (default), the data is returned as raw.
    :param lighting_type:
        Lighting type either 'sun' or 'sky'. This is used to get the default sun or sky file if
        `file_name` was not given.
    :return:
        (wls, irradiances) tuple where wls is a list of wavelengths (bands) and irradiances are
        corresponding list of irradiances. The length of the lists vary depending on given bandwith.
    :raises
        ValueError if `file_name` was not provided and `type` is not either 'sun' or 'sky'.
    """

    if not file_name:
        if lighting_type == 'sun':
            file_name = C.file_default_sun
        elif lighting_type == 'sky':
            file_name = C.file_default_sky
        else:
            raise ValueError(f"Lighting file name was not provided. For loading one of the default files, "
                             f"expected file type either 'sun' or 'sky', was '{lighting_type}'.")

    path = _find_lighting_file(file_name, forest_id, lighting_type=lighting_type)

    wls, irradiances, _ = read_psg_file(path)

    if sampling is not None:
        new_irradiances = SU.resample(original_wl=wls, original_val=irradiances, new_wl=sampling)
        wls = sampling
        irradiances = new_irradiances

    return np.array(wls), np.array(irradiances)


def _find_lighting_file(file_name: str, forest_id: str = None, lighting_type: str = 'sun') -> str:
    """Attempts to find a lighting file with given filename.

    :param file_name:
        A file with this name is searched from sun_data directory. If
        also scene_id is given, the search is extended to scene directory. Precedence
        is then for the file in the scene directory.
    :param forest_id:
        Optional. If not given, forest scene directory is not searched.
    :return:
        Path to found file.
    :raises FileNotFoundError:
        if the file is not found.
    """

    if forest_id is not None:
        logging.info(f"Trying to find lighting data from forest scene directory '{PH.path_directory_forest_scene(forest_id)}'.")
        p = PH.join(PH.path_directory_forest_scene(forest_id), file_name)
        if os.path.exists(p):
            logging.info(f"Lighting data found.")
            return p
        else:
            logging.info(f"Could not find sun data from scene directory.")

    if lighting_type == 'sun':
        p_dir = PH.path_directory_sun_data()
    elif lighting_type == 'sky':
        p_dir = PH.path_directory_sky_data()
    else:
        raise ValueError(f"For searching from one of the default lighting directories, "
                         f"expected type either 'sun' or 'sky', was '{lighting_type}'.")

    logging.info(f"Trying to find lighting data from '{p_dir}'.")
    p = PH.join(p_dir, file_name)
    if os.path.exists(p):
        logging.info(f"Lighting data found.")
        return p

    raise FileNotFoundError(f"File with name '{file_name}' was not found.")


if __name__ == '__main__':

    """
    This main can be used for testing.
    
    Game plan:
        1. find sun file
        2. read file and check if ok
        3. if not, fix spaces and save
        4. if irradiance not in [W/m^2/nm] read, fix units and save
        5. read file to memory
        6. integrate over bandwith 
        7. return bands and irradiances
    """
    import sys
    from src import plotter
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # load_sun('ASTM_G173-03.csv', bandwith=1, spectral_range=(400,2500))
    # load_sun('psg_rad.txt', bandwith=1, spectral_range=(400,2500))
    bandwith = 100
    sunfile = 'default_sun.txt'
    # wls, irradiances = load_sun(sunfile)
    wls_b, irradiances_b = load_light()
    last = irradiances_b[-1]
    plotter.plot_light_data(wls_b, irradiances_b, forest_id="0102231033", sun_plot_name='default_sun')
    # plotter.plot_sun_data(wls, irradiances, scene_id="0123456789", sun_filename=sunfile, show=True)
    print('m')

    # TODO check and fix spectral range from 400 to 2500

    # Test binning
    # s_range = [10,30]
    # bandwith = 20
    # wls = list(range(s_range[0], s_range[1], 5))
    # wls = [1, 1.5, 2, 2.5, 3, 4, 5,6]
    # irradiances = np.ones_like(wls) * 5
    # irradiances = [2,1,2,5,40,3,2,3.1]
    # wls, irradiances = check_resolution(wls, irradiances)

    # wls_binned, irradiances_binned = bin(wls, irradiances, bandwith)
    # print(wls_binned)
    # print(irradiances_binned)

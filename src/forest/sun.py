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
import csv
import os
import logging
import math
import matplotlib.pyplot as plt

from src.data import path_handling as PH


def is_resolution_1nm(wls):
    """Check that the wavelength resolution is 1 nm. """

    a = wls[0]
    biggest_diff = 0
    smallest_diff = 1e10

    for i in range(len(wls)-1):
        diff = math.fabs(a-wls[i+1])
        a = wls[i+1]
        if diff > biggest_diff:
            biggest_diff = diff
        if smallest_diff > diff:
            smallest_diff = diff

    if math.fabs(1.0 - biggest_diff) > 0.01 or math.fabs(1.0 - smallest_diff) > 0.01:
        return False
    else:
        return True


def integrate(wls, irradiances, bandwith):
    """Integrate irradiances to given bandwith.

    If given bandwith is 10, the first 10 irradiances (in 1 nm resolution file) is
    summed to 400 nm band. Then the next 10 are summed to 410 nm band, and so on.

    Hard-coded to start from 400 nm.

    :param wls:
        Original wavelengths.
    :param irradiances:
        Irradiances corresponding to wls.
    :param bandwith:
        Bandwith of integration.
    :returns:
        (wls, irradiances) tuple where wls is a list of new integrated wavelengths and
        irradiances is a list of corresponding irradiances.
    """

    wls_binned = [wls[0]]
    irradiances_binned = []

    start_wl = 400
    stop_wl = start_wl + bandwith
    sum = 0

    i_bin = 0
    for i, wl in enumerate(wls):
        if wl >= start_wl and wl < stop_wl:
            sum += irradiances[i]
        else:
            start_wl = stop_wl
            stop_wl = stop_wl + bandwith
            irradiances_binned.append(sum)
            sum = irradiances[i]
            wls_binned.append(wls_binned[i_bin]+bandwith)
            i_bin += 1
        if i == len(wls)-1:
            # wls_binned.append(wls[-1])
            irradiances_binned.append(sum)
            # The lists have been exhausted. Add what is remaining in the last bin
            # Note that the first and last bin does not necessarily have the same amount of summed elements

    return wls_binned, irradiances_binned


def read_sun_data(path):
    """Read sun data and return wavelengths, irradiances and comments.
    :returns:
        (wls, irradiances, comments) tuple where wls and irradiances are numpy arrays.
        comments is a list of comment rows needed in file fixing.
    """

    logging.info(f"Reading sun data.")
    wls = []
    irradiances = []
    comments = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if row[0].startswith('#'):
                comments.append(row)
                continue
            else:
                wls.append(float(row[0]))
                irradiances.append(float(row[1]))
                # skip the rest of the data fields

    resolution_ok = is_resolution_1nm(wls)
    if not resolution_ok:
        raise RuntimeError(f"Resolution in given file is not 1 nm. Use default sun or fix the file.")

    return np.array(wls), np.array(irradiances), comments


def fix_double_space(path: str):
    """Replace double spaces with a single space."""

    with open(path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('  ', ' ')

    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)


def fix_nasa_sun(path:str):
    """ Fix double spaces and irradiance units provided by NASA.

    Rewrites the file (several times) if needed. Adds a tag to the fixed file that
    tells it is now OK.
    """

    with open(path, 'r') as file:
        filedata = file.read()
        if filedata.startswith('# HyperBlend compliance'):
            logging.info(f"Given sun file OK.")
            return
        else:
            logging.info(f"Trying to convert given sun file to HyperBlend compilable.")

    fix_double_space(path)

    wls_i, irradiances_i, comments = read_sun_data(p)
    unit_row_idx = 0
    unit_row_content = []
    for i,comment in enumerate(comments):
        if 'Radiance' == comment[1] and 'unit:' == comment[2]:
            unit = comment[-1]
            print(unit)
            unit_row_idx = i

            if unit == '[W/m2/nm]':
                print(f"Irradiance unit is [W/m2/nm], all good.")
                unit_row_content = comment
            elif unit == '[W/m2/um]':
                print(f"Irradiance unit is [W/m2/um], converting to [W/m2/nm].")
                irradiances_i = irradiances_i * 0.001

                for piece in comment:
                    if piece != unit:
                        unit_row_content.append(piece)
                    else:
                        unit_row_content.append('[W/m2/nm]')
            else:
                raise RuntimeError(f"Cannot convert unit '{unit}' to '[W/m2/nm]'. Use [W/m2/um] when downloading a file.")

    comments[unit_row_idx] = unit_row_content
    comments.insert(0, ['#', 'HyperBlend', 'compliance'])

    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(comments)
        writer.writerows(list(zip(wls_i, irradiances_i)))


def find_file(file_name: str, scene_id=None) -> str:
    """Attempts to find a sun file with given filename.

    :param file_name:
        A file with this name is searched from sun_data directory. If
        also scene_id is given, the search is extended to scene directory. Precedence
        is then for the file in the scene directory.
    :param scene_id:
        Optional. If not given, scene directory is not searched.
    :return:
        Path to found file.
    :raises FileNotFoundError:
        if the file is not found.
    """

    if scene_id:
        logging.info(f"Trying to find sun data from scene directory '{PH.path_directory_forest_scene(scene_id)}'.")
        if os.path.exists(PH.join(PH.path_directory_forest_scene(scene_id), file_name)):
            logging.info(f"Sun data found.")
            p = PH.join(PH.path_directory_forest_scene(scene_id), file_name)
            return p
        else:
            logging.info(f"Could not find sun data from scene directory.")

    logging.info(f"Trying to find sun data from '{PH.path_directory_sun_data()}'.")
    if os.path.exists(PH.join(PH.path_directory_sun_data(), file_name)):
        logging.info(f"Sun data found.")
        p = PH.join(PH.path_directory_sun_data(), file_name)
        return p

    raise FileNotFoundError(f"File with name '{file_name}' was not found.")


def load_sun(file_name: str = None, scene_id=None, bandwith=1):
    """ Loads a sun file and returns wavelengths and corresponding irradiances.

    If given file is not compatible with HyperBlend, an attempt is made to fix it.
    Given file must contain wavelengths from 400 to 2500 (both inclusive) and have spectral
    resolution of 1 nm.

    :param file_name:
        Optional. If given, this file is searched from sun_data directory. If
        also scene_id is given, the search is extended to scene directory. Precedence
        is then for the file in the scene directory. If not given, the default sun file
        from the repository is used.
    :param scene_id:
        If scene_id is given, the search is extended to scene directory so that the scene
        directory is searched first and if found, it is returned. If not found, the sun_data
        directory is then searched.
    :param bandwith:
        Optional. Bandwith of integration, i.e., if given bandwith is 10, the first 10 irradiances (in 1 nm
        resolution file) is summed to 400 nm band. Then the next 10 are summed to 410 nm band, and so on.
        Default is 1.
    :return:
        (wls, irradiances) tuple where wls is a list of wavelengths (bands) and irradiances are
        corresponding list of irradiances. The length of the lists vary depending on given bandwith.
    """

    if not file_name:
        file_name = "default_sun.txt"
    path = find_file(file_name, scene_id)
    fix_nasa_sun(path)
    wls, irradiances, _ = read_sun_data(path)
    if bandwith != 1:
        wls, irradiances = integrate(wls, irradiances, bandwith)
    return wls, irradiances


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
    wls_b, irradiances_b = load_sun(file_name=sunfile, bandwith=bandwith)
    last = irradiances_b[-1]
    plotter.plot_sun_data(wls_b, irradiances_b, scene_id="0123456789", sun_filename='default_sun')
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

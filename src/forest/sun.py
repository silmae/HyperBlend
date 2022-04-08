"""
Sun-related stuff
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
    """Integrate irradiances to given bandwith."""

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
    """Read sun data and return wavelngths, irradiances and comments."""

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


    print('sdf')
    return np.array(wls), np.array(irradiances), comments


def plot_sun_data(wls, irradiances, wls_binned=None, irradiances_binned=None):
    plt.plot(wls, irradiances, label='Sun 1 nm')
    if wls_binned and irradiances_binned:
        plt.plot(wls_binned, irradiances_binned, label='Binned', alpha=0.5)
    plt.legend()
    plt.show()


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
    """ Fix double spaces and irradiance units provided by NASA."""

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


def find_file(file_name: str, scene_id=None):
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


def load_sun(file_name: str, scene_id=None, bandwith=1, spectral_range=(400,2500)):
    path = find_file(file_name, scene_id)
    fix_nasa_sun(path)
    wls, irradiances, _ = read_sun_data(path)
    if bandwith != 1:
        wls, irradiances = integrate(wls, irradiances, bandwith)
    return wls, irradiances


if __name__ == '__main__':

    """
    Game plan:
        1. if not at 1nm resolution, interpolate
        2. extrapolate ends with constant value
        3. sum over the shit
    """
    import sys
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # load_sun('ASTM_G173-03.csv', bandwith=1, spectral_range=(400,2500))
    # load_sun('psg_rad.txt', bandwith=1, spectral_range=(400,2500))
    wls, irradiances = load_sun('psg_rad.txt')
    wls_b, irradiances_b = load_sun('psg_rad.txt', bandwith=10)
    last = irradiances_b[-1]
    plot_sun_data(wls, irradiances, wls_b, irradiances_b)
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

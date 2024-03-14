import csv
import logging
import math
import os

import numpy as np

from src.data import path_handling as PH


def read_light_file(path:str):

    comments = _load_file_comments_only(path)
    is_psg = False
    is_goa = False
    for line in comments:
        for token in line:
            if 'PSG' in token:
                is_psg = True
                break
            elif 'GOA' in token:
                is_goa = True
                break

    if is_psg:
        print(f"This is a PSG file")
        wls, irradiances,_ = read_psg_file(path, try_to_fix=True)
    if is_goa:
        print(f"This is a GOA file")
        read_goa_file(path)
        first_line = _load_first_non_comment_line(path)
        if len(first_line) > 2:
            logging.info(f"Splitting GOA-generated file into separate sun and sky files.")
            split_goa_file(path)

        wls, irradiances = read_goa_file(path)


    return np.array(wls), np.array(irradiances)



def split_goa_file(path):

    comments = []
    wls = []
    direct_irradiances = []
    diffuse_irradiances = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if row[0].startswith('#'):
                comments.append(row)
                continue
            else:
                wls.append(float(row[0]))
                direct_irradiances.append(float(row[2]))
                diffuse_irradiances.append(float(row[3]))


    dir_path = os.path.dirname(path)
    base_file_name = os.path.basename(path)
    new_sun_file_name = base_file_name + "_sun.txt"
    new_sky_file_name = base_file_name + "_sky.txt"
    out_path_sun = PH.join(dir_path, new_sun_file_name)
    out_path_sky = PH.join(dir_path, new_sky_file_name)

    write_hb_light_file(path=out_path_sun, wls=wls, irradiances=direct_irradiances, comments=comments)
    write_hb_light_file(path=out_path_sky, wls=wls, irradiances=diffuse_irradiances, comments=comments)


def read_goa_file(path):
    # TODO this could be the main "read sensible hyperblend compliant file"
    #   and the file fixing is done separately.
    pass


def read_psg_file(path, try_to_fix=True):
    """Read NASA Planetary Spectrum Generator file and return wavelengths, irradiances and comments.

    :param path:
        Path to a PSG file.
    :param try_to_fix:
        If True (default), an attempt is made to fix the file to a more easier for for a csv reader to read.
        Otherwise, this step is skipped.
    :returns:
        (wls, irradiances, comments) tuple where wls and irradiances are numpy arrays.
        comments is a list of comment rows needed in file fixing.
    :raises
        RuntimeError if wavelengths in the file are not in 1 nm resolution.
    """


    if try_to_fix:
        logging.info("Trying to fix PSG generated file.")
        _fix_psg_file(path)

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

    epsilon = 0.01
    min_diff, max_diff = _band_separations(wls)
    resolution_ok = abs(min_diff-1) < epsilon and abs(max_diff - 1) < epsilon
    if not resolution_ok:
        raise RuntimeError(f"Resolution in given file is not 1 nm. Use default sun or fix the file.")

    return np.array(wls), np.array(irradiances), comments


def _load_file_comments_only(path: str):

    comments = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if row[0].startswith('#'):
                comments.append(row)
                continue
    return comments


def _load_first_non_comment_line(path: str):

    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if row[0].startswith('#'):
                continue
            else:
                return row


def _band_separations(wls):
    """Minimum and maximum of adjacent wavelengths in given list.

    :param wls:
        List of wavelengths (floats).
    :return:
        min_diff, max_diff (float) minimum and maximum differences.
    """

    previous_wl = wls[0]
    max_diff = 0
    min_diff = 1e10

    for i in range(len(wls)-1):
        current_wl = wls[i+1]
        diff = math.fabs(current_wl-previous_wl)
        previous_wl = current_wl
        if diff > max_diff:
            max_diff = diff
        if diff < min_diff:
            min_diff = diff

    return min_diff, max_diff


def _fix_psg_file(path: str):
    """ Fix double spaces and irradiance units provided by NASA PSG.

    Rewrites the file (several times) if needed. Adds a tag to the fixed file that
    tells it is now OK.
    """

    with open(path, 'r') as file:
        filedata = file.read()
        if filedata.startswith('# HyperBlend compliance'):
            logging.info(f"Light file OK.")
            return
        else:
            logging.info(f"Trying fix light file for easier handling.")

    _fix_double_space(path)

    # This would be circular call if not for try_to_fix=False
    wls_i, irradiances_i, comments = read_psg_file(path, try_to_fix=False)
    unit_row_idx = 0
    unit_row_content = []
    for i,comment in enumerate(comments):
        if 'Radiance' == comment[1] and 'unit:' == comment[2]:
            unit = comment[-1]
            unit_row_idx = i

            if unit == '[W/m2/nm]':
                logging.info(f"Irradiance unit is [W/m2/nm], all good.")
                unit_row_content = comment
            elif unit == '[W/m2/um]':
                logging.info(f"Irradiance unit is [W/m2/um], converting to [W/m2/nm].")
                irradiances_i = irradiances_i * 0.001

                for piece in comment:
                    if piece != unit:
                        unit_row_content.append(piece)
                    else:
                        unit_row_content.append('[W/m2/nm]')
            else:
                raise RuntimeError(f"Cannot convert unit '{unit}' to '[W/m2/nm]'. Use [W/m2/um] when generating spectra.")

    comments[unit_row_idx] = unit_row_content
    comments.insert(0, ['#', 'HyperBlend', 'compliance'])

    write_hb_light_file(path=path, wls=wls_i, irradiances=irradiances_i, comments=comments)


def write_hb_light_file(path, wls, irradiances, comments):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(comments)
        writer.writerows(list(zip(wls, irradiances)))


def _fix_double_space(path: str):
    """Replace double spaces with a single space."""

    with open(path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('  ', ' ')

    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)

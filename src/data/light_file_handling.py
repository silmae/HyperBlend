"""

This script contains methods for reading and writing light spectrum files.

Files formatted so that comment lines are prefixed with '#' and rest of the lines
contain wavelength-irradiance pairs that can be casted to floats, they can be read
directly. For spectra generated with either NASA PSG or SSolar GOA, the files are
fixed so that they can be read directly later.

How to generate sun spectra with NASA PSG
-----

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


How to generate sun and sky spectra with SSolar GOA
-----

    1. Download Windows version of GOA from https://goa.uva.es/ssolar_goa-model/ or use their
        GitHub version from https://github.com/GOA-UVa/SSolar-GOA.
    2. (Assuming the Windows version) Click on "Load default settings"
    3. Change wavelength range to 400-2500 nm
    4. Click on "Run"
    5. Move the generated file to where HyperBLend's light files are stored and
        call read_light_file() in this script. It will the split the sun and sky
        spectra to separate files that can be read by calling the read_light_file()
        again.
"""

import csv
import logging
import math
import os

import numpy as np

from src.data import path_handling as PH


def read_light_file(path: str):

    comments = _read_file_comments_only(path)
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

    if not is_psg and not is_goa:
        logging.warning(f"Unknown light file type. If it is formatted so that comment lines are "
                        f"prefixed with '#' and rest of the lines contain wavelenght-irradiance pairs "
                        f"that can be casted to floats, everything should go well.")

    if is_psg:
        _fix_psg_file(path)
        wls, irradiances, _ = _read_hb_light_file(path)
    if is_goa:
        first_line = _read_first_non_comment_line(path)
        if len(first_line) > 2:
            logging.info(f"Splitting GOA-generated file into separate sun and sky files.")
            # This is all cool and dandy but how do we know which file the caller wants?
            new_sun_file_name, new_sky_file_name = _split_goa_file(path)
            raise RuntimeError(f"The GOA-generated file was not yet split into separate sun "
                               f"and sky spectrum files. I splitted them for you, but since I "
                               f"don't know which one you want, you must call me again by one of the "
                               f"new file names: sun file '{new_sun_file_name}' or sky file '{new_sky_file_name}'. "
                               f"So this is not an error as such, but I have to halt the execution to avoid "
                               f"problems for you.")

        wls, irradiances, _ = _read_hb_light_file(path)

    return np.array(wls), np.array(irradiances)


def _fix_psg_file(path: str):
    """ Fix double spaces and irradiance units provided by NASA PSG.

    Rewrites the file (several times) if needed. Adds a tag to the fixed file that
    tells it is now OK.
    """

    with open(path, 'r') as file:
        filedata = file.read()
        if filedata.startswith('# HyperBlend compliance'):
            logging.info(f"PSG-generated light spectrum file OK.")
            return
        else:
            logging.info(f"Trying fix light file for easier handling.")

    _fix_double_space(path)

    # This would be circular call if not for try_to_fix=False
    wls_i, irradiances_i, comments = _read_hb_light_file(path, try_to_fix=False)
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

    _write_hb_light_file(path=path, wls=wls_i, irradiances=irradiances_i, comments=comments)
    logging.info(f"PSG-generated light file should now be fixed.")


def _split_goa_file(path):
    """Splits GOA-generated spectrum file into separate sun and sky spectra.

    :param path:
        Path to the GOA-generated file.
    :return:
        new_sun_file_name, new_sky_file_name Names of the new files.
    """

    logging.info("Splitting GOA-generated spectral light file.")
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

    # Fix the header comment line
    new_comments = []
    for comment in comments:
        if 'Wavelength' in comment:
            new_comments.append(["#", "Wavelength", "Irradiance"])
        else:
            new_comments.append(comment)

    dir_path = os.path.dirname(path)
    base_file_name = os.path.basename(path).rsplit('.')[0]
    new_sun_file_name = base_file_name + "_sun.txt"
    new_sky_file_name = base_file_name + "_sky.txt"
    out_path_sun = PH.join(dir_path, new_sun_file_name)
    out_path_sky = PH.join(dir_path, new_sky_file_name)

    _write_hb_light_file(path=out_path_sun, wls=wls, irradiances=direct_irradiances, comments=new_comments)
    _write_hb_light_file(path=out_path_sky, wls=wls, irradiances=diffuse_irradiances, comments=new_comments)

    logging.info(f"Splitting done. You can find the new sun file from '{out_path_sun}' and the "
                 f"sky file from '{out_path_sky}'.")

    return new_sun_file_name, new_sky_file_name


def _write_hb_light_file(path: str, wls, irradiances, comments):
    """Writes a light spectrum file.

    The produced file will start with comments where each line is
    prefixed with '#' and the rest of the lines will contain wls-irradiance
    pair separated by a space.

    :param path:
        Path where to write the file.
    :param wls:
        Wavelengths as a list of floats.
    :param irradiances:
        Associated irradiances as a list of floats.
    :param comments:
        Comment lines as a list of strings prefixed with '#'.
    """

    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(comments)
        writer.writerows(list(zip(wls, irradiances)))


def _read_hb_light_file(path, required_resolution=1.0, resolution_epsilon=0.01):
    """Read light spectrum from a file of certain structure.

    This method can read the files that are fixed by other methods to be compliant
    with HyperBlend expectations.

    The file is supposed to start with rows where the first character is '#'.
    These lines are considered comments. The remaining lines must have to
    numbers separated by a space. The numbers have to castable to a float.

    Example of how the file could look like:

    |  # This is a comment line that can contain some metadata.
    |  # Perhaps this line will tell that the power is in W/m^2/nm
    |  # Maybe some header data like "Wavelengths Irradiance" and then the actual data in following lines
    |  400.0 0.12345
    |  401.0 0.6789
    |  ...

    :param path:
        Path to a file to be read.
    :param required_resolution:
        Spectral resolution of the file is checked so that bands cannot be further
        away from each other than this. Default is 1.0.
    :param resolution_epsilon:
        Allow this much variation between adjacent bands. Default is 0.01.
    :returns:
        (wls, irradiances, comments) tuple where wls and irradiances are numpy arrays.
        comments is a list of comment rows needed in file fixing. The comments that are
        returned are mainly for internal working of this script and do not matter for outside
        caller.
    :raises
        RuntimeError if wavelengths in the file are not in 1 nm resolution.
    """

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

    min_diff, max_diff = _band_separations(wls)
    resolution_ok = abs(min_diff-required_resolution) < resolution_epsilon and abs(max_diff-required_resolution) < resolution_epsilon
    if not resolution_ok:
        raise RuntimeError(f"Resolution in given file is not 1 nm. Use default sun or fix the file.")

    return np.array(wls), np.array(irradiances), comments


def _read_file_comments_only(path: str):
    """Reads only comment lines beginning with '#' and returns those lines as a list."""

    comments = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if row[0].startswith('#'):
                comments.append(row)
                continue
    return comments


def _read_first_non_comment_line(path: str):
    """Reads only the first line after comments and returns that."""

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


def _fix_double_space(path: str):
    """Replace double spaces with a single space."""

    with open(path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('  ', ' ')

    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)

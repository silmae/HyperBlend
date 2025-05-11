"""
This file contains methods for creating target reflectances and transmittances in
a form that the rest of the project code can use it.

Some default targets can be created without fetching external data.

"""

import numpy as np
import logging

# SpectRes is used for resampling spectra to lower resolution

from src import constants as C
from scipy.interpolate import CubicSpline


VIS_MIN = 380.0
VIS_MAX = 700.0


def spectra_to_rgb(wls, value):
    """Creates at least some sort of RGB presentation of given spectrum.

    Expects `value`s to be reflectance no greater than 1. If greater values are found,
    the whole `value` list is normalized to 1.

    If wavelengths are completely outside of visible range the edges and the middle point of
    the spectrum are used. If they are at least partially inside the visible spectrum, closest
    values to red, green and blue are returned. There is no guarantee that these are different
    values, e.g., blue and green may be the same value if not enough wavelengths exist in visible
    range. If only 3 wavelengths are provided they are returned. For less than 3 wavelengths,
    a default color is returned.

    :param wls: Wavelengths as a list of floats.
    :param value: Reflectance values as a list of floats.
    :return: Returns an rgb value as a list of floats.
    :raises ValueError: If given lists are different in length.
    """

    n = len(wls)
    if n != len(value):
        raise ValueError(
            f"Given wavelength and value lists are different length {n} != {len(value)}."
        )

    default_rgb = [0.7, 0.3, 0.6]

    wls_a = np.array(wls)
    value_a = np.array(value)
    max_value = np.max(value_a)

    # Normalize to 1 if values greater than 1 were found because RGB in Blender ranges from 0 to 1.
    if max_value > 1:
        value_a = value_a / max_value

    if n < 3:
        logging.info(
            f"Length of provided wavelength list less than 3. Returning default color '{default_rgb}'."
        )
        return default_rgb
    elif n == 3:
        logging.info(
            f"Length of provided wavelength is 3. Returning original values normalized to 1."
        )
        return value_a
    else:
        if is_in_visible(wls_a):
            b = value_a[find_nearest_idx(wls_a, C.default_B_wl)]
            g = value_a[find_nearest_idx(wls_a, C.default_G_wl)]
            r = value_a[find_nearest_idx(wls_a, C.default_R_wl)]
            res = [r, g, b]
            logging.info(
                f"Values in visible range. Returning approximate false color representation '{res}'."
            )
        else:
            b = value_a[0]
            g = value_a[int(n / 2)]
            r = value_a[-1]
            res = [r, g, b]
            logging.info(
                f"Values not in visible range. Returning some sort of false color representation "
                f"using edges and middle values '{res}'."
            )
        return res


def is_in_visible(wls):
    """Check if given wavelengths are on visible range.

    :param wls: List of wavelengths to check.
    :return: False if all wavelengths are either greater than 700 nm or lower than 380 nm.
        Otherwise, returns True, i.e., it is enough that even some part of the spectrum is
        on the visible.
    """

    wls_a = np.array(wls)

    if np.all(np.greater(wls_a, np.ones_like(wls_a) * VIS_MAX)):
        return False

    if np.all(np.less(wls_a, np.ones_like(wls_a) * VIS_MIN)):
        return False

    return True


def find_nearest_idx(array, value):
    """Find nearest index of value in array.

    :param array: Array to be searched for.
    :param value: Value that is searched.
    :return: Index of an element that is closest to `value`.
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def resample(original_wl, original_val, new_wl):
    """Resample spectra to lower resolution.

    Github page for the SpectRes package: https://github.com/ACCarnall/SpectRes
    And documentation: https://spectres.readthedocs.io/

    :param original_wl: List or 1-D numpy array of wavelengths in the original spectrum.
    :param original_val: List or numpy array of intensities corresponding to `original_wl`.
        If multidimensional numpy array is provided, the last dimension
        must match the length of `original_wl`. Other dimensions can be used
        to include multiple spectra at the same run.
    :param new_wl: List or 1-D numpy array of new wavelengths to be resampled to.
    :return: Returns resampled spectra as a numpy array. The first dimension is the
        same length as `new_wl` and other dimensions the same as in `original_val`.
    """

    # Cast to numpy arrays in case they were not already
    new_wavs = np.array(new_wl)
    spec_wavs = np.array(original_wl)
    spec_fluxes = np.array(original_val)

    # Disabled spectres resampling because it behaves badly at the ends of data
    # resampled = spectres.spectres(new_wavs=new_wavs, spec_wavs=spec_wavs, spec_fluxes=spec_fluxes, fill=0.0)

    # Instead, use Scipy cubic interpolation
    if len(spec_fluxes.shape) > 1:  # ref and tran given together
        cs_ref = CubicSpline(spec_wavs, spec_fluxes[0])  # reflectance
        cs_tran = CubicSpline(spec_wavs, spec_fluxes[1])  # transmittance
        resampled = np.zeros((2, len(new_wavs)))
        resampled[0] = cs_ref(new_wavs)
        resampled[1] = cs_tran(new_wavs)
    else:  # some general 1D spectra to interpolate
        cs = CubicSpline(spec_wavs, spec_fluxes)
        resampled = cs(new_wavs)
    return resampled

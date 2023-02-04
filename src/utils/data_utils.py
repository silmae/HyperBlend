"""
Reading and manipulating image data.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.data import file_handling as FH
from src import constants as C


def get_relative_refl_or_tran(imaging_type: str, wl: float, base_path: str) -> float:
    """Returns leaf reflectance (or transmittance) divided by reference reflectance (or transmittance).

    :param imaging_type:
        String, either 'refl' for reflectance or 'tran' for transmittance.
    :param wl:
        Wavelength.
    :param base_path:
        Top level path (working temp directory).
    :return:
        Relative reflectance or transmittance as a single float.
    """

    leaf_mean = get_rend_as_mean(FH.search_by_wl(C.target_type_leaf, imaging_type, wl, base_path))
    reference_mean = get_rend_as_mean(FH.search_by_wl(C.target_type_ref, imaging_type, wl, base_path))
    relative = leaf_mean / reference_mean
    return relative


def get_rend_as_mean(image_file_path: str) -> float:
    """Read image from given path and return mean of the pixels."""

    array = get_rend_as_ndarray_wl(image_file_path)
    return np.mean(array)


def get_rend_as_ndarray_wl(image_file_path: str):
    """ Read an image in given path into a Numpy array."""

    if os.path.exists(image_file_path):
        array = plt.imread(image_file_path)
        return array
    else:
        raise Exception(f"Image {image_file_path} does not exist.")


def unpack_target(target):
    """Unpacks target as given by toml_handling into separate lists.

    toml_handling.read_target returns wavelengths, reflectances and transmittances
    as a list of tuples. In many cases it is beneficial to handle them as three separate lists.
    This is reverse of pack_target().

    :param target:
        Target to be unpacked.
    :return:
        3 lists: wls, refls, trans
    """

    wls, refls, trans = map(list, zip(*target))
    return wls, refls, trans


def pack_target(wls, refls, trans):
    """Pack lists of wavelengths, reflectances and transmittances into a numpy array of tuples.

    This is reverse of unpack_target(). This method can be used to pack lists
    of data in a form that can be passed to toml_handling.write_target() method.

    :param wls:
        List of wavelengths.
    :param refls:
        List of reflectances.
    :param trans:
        List of transmittances.
    :return:
        Numpy array of tuples (wl, reflectance, transmittance).
    """

    target = np.zeros((3, len(wls)))
    target[0] = wls
    target[1] = refls
    target[2] = trans
    target = np.transpose(target)
    return target

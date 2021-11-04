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

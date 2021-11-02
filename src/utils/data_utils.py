import os

import numpy as np
import matplotlib.pyplot as plt

from src.data import file_handling as FH
from src import constants as C


def get_relative_refl_or_tran_series(imaging_type: str, wl_list, base_path: str):
    res = []
    for _,wl in enumerate(wl_list):
        val = np.mean(plt.imread(FH.search_by_wl(C.target_type_leaf, imaging_type, wl, base_path)))
        ref = np.mean(plt.imread(FH.search_by_wl(C.target_type_ref, imaging_type, wl, base_path)))
        res.append(val / ref)
    return np.array(res)


def get_relative_refl_or_tran(imaging_type: str, wl: float, base_path: str):
    """Returns leaf reflectance (transmittance) divided by reference reflectance (transmittance). """

    leaf_mean = get_rend_as_mean(FH.search_by_wl(C.target_type_leaf, imaging_type, wl, base_path))
    reference_mean = get_rend_as_mean(FH.search_by_wl(C.target_type_ref, imaging_type, wl, base_path))
    relative = leaf_mean / reference_mean
    return relative


def get_rend_as_mean(image_file_path: os.path) -> float:
    array = get_rend_as_ndarray_wl(image_file_path)
    return np.mean(array)


def get_rend_as_ndarray_wl(image_file_path: os.path):
    if os.path.exists(image_file_path):
        array = plt.imread(image_file_path)
        return array
    else:
        raise Exception(f"Image {image_file_path} does not exist.")

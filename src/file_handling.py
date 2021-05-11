
import os

from src import constants as C
from src import utils

def check_folder_structure():
    pass


def get_leaf_rend_folder() -> os.path:
    """Returns the path to leaf render folder."""

    rend_path_leaf = C.project_root_path + '/' + C.rend_folder_name
    return os.path.normpath(rend_path_leaf)


def get_reference_rend_folder(imaging_type: str) -> os.path:
    """Returns the path to reflectance or transmittance reference folder."""

    if imaging_type == C.imaging_type_refl:
        rend_path_refl_ref = C.project_root_path + '/' + C.rend_ref_refl_folder_name
        return os.path.normpath(rend_path_refl_ref)
    elif imaging_type == C.imaging_type_tran:
        rend_path_tran_ref = C.project_root_path + '/' + C.rend_ref_tran_folder_name
        return os.path.normpath(rend_path_tran_ref)
    else:
        raise Exception(f"Imaging type {imaging_type} not recognized. Use {C.imaging_type_refl} or {C.imaging_type_tran}.")


def get_image_folder(target_type: str, imaging_type: str):
    """Returns a path to correct folder according to given target and imaging type. """

    if target_type == C.target_type_leaf:
        return get_leaf_rend_folder()
    elif target_type == C.target_type_ref:
        return get_reference_rend_folder(imaging_type)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_leaf}. Was {target_type}.")


def get_image_file_path(target_type: str, imaging_type: str, wl: float):
    """Returns a full path to an image of given wavelength."""

    image_name = f"{imaging_type}_wl{wl:.2f}.tif"
    if target_type == C.target_type_leaf:
        return os.path.normpath(get_leaf_rend_folder() + '/' + image_name)
    elif target_type == C.target_type_ref:
        return os.path.normpath(get_reference_rend_folder(imaging_type) + '/' + image_name)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_leaf}. Was {target_type}.")


def search_by_wl(target_type: str, imaging_type: str, wl: float) -> os.path:
    """Search a folder for an image of given wavelength.

    A path to the image is returned.

    :raises FileNotFoundError if not found
    """

    def almost_equals(f1: float, f2: float, epsilon=0.01):
        res = abs(f1 - f2) <= epsilon
        return res

    folder = get_image_folder(target_type, imaging_type)
    for filename in os.listdir(folder):
        image_wl = utils.parse_wl_from_image_name(filename)
        if almost_equals(wl, image_wl):
            return get_image_file_path(target_type,imaging_type,wl)

    # Did not find
    raise FileNotFoundError(f"Could not find {wl} nm image from {folder}.")

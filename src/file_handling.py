import logging
import os

from src import constants as C
from src import utils


def create_opt_folder_structure(set_name: str):
    """Check that the folder structure for optimization is OK. Create if not."""
    if not os.path.exists(get_path_opt_root()):
        os.makedirs(get_path_opt_root())
    if not os.path.exists(get_path_opt_target(set_name)):
        os.makedirs(get_path_opt_target(set_name))
    if not os.path.exists(get_path_opt_working(set_name)):
        os.makedirs(get_path_opt_working(set_name))
    if not os.path.exists(get_path_rend_leaf(set_name)):
        os.makedirs(get_path_rend_leaf(set_name))
    if not os.path.exists(get_path_rend_reference(C.imaging_type_refl, get_path_opt_working(set_name))):
        os.makedirs(get_path_rend_reference(C.imaging_type_refl, get_path_opt_working(set_name)))
    if not os.path.exists(get_path_rend_reference(C.imaging_type_tran, get_path_opt_working(set_name))):
        os.makedirs(get_path_rend_reference(C.imaging_type_tran, get_path_opt_working(set_name)))
    if not os.path.exists(get_path_opt_result(set_name)):
        os.makedirs(get_path_opt_result(set_name))
    if not os.path.exists(get_path_opt_result_plot(set_name)):
        os.makedirs(get_path_opt_result_plot(set_name))
    if not os.path.exists(get_path_opt_subresult(set_name)):
        os.makedirs(get_path_opt_subresult(set_name))


def get_path_opt_root():
    p = os.path.normpath(C.path_project_root + '/' + C.folder_opt)
    return p


def get_path_opt_set(set_name: str):
    p = os.path.normpath(get_path_opt_root() + '/' + set_name)
    return p


def get_path_opt_target(set_name: str):
    p = os.path.normpath(get_path_opt_set(set_name) + '/' + C.folder_opt_target)
    return p


def get_path_opt_working(set_name: str):
    p = os.path.normpath(get_path_opt_set(set_name) + '/' + C.folder_opt_work)
    return p


def get_path_opt_result(set_name: str):
    p = os.path.normpath(get_path_opt_set(set_name) + '/' + C.folder_opt_result)
    return p


def get_path_opt_result_plot(set_name: str):
    p = os.path.normpath(get_path_opt_result(set_name) + '/' + C.folder_opt_plot)
    return p


def get_path_opt_subresult(set_name: str):
    p = os.path.normpath(get_path_opt_result(set_name) + '/' + C.folder_opt_subresult)
    return p


def subresult_exists(set_name: str, wl:float):
    # FIXME this is the same that is used by tom_handler when creating the file. Collect to same place.
    p = get_path_opt_subresult(set_name) + '/' + f"subres_wl_{wl:.2f}" + C.postfix_text_data_format
    if os.path.exists(p):
        return True
    else:
        return False


def target_exists(set_name: str) -> bool:
    return os.path.exists(get_path_opt_set(set_name))


def get_path_rend_leaf(set_name: str):
    """Returns the path to leaf render folder."""
    p = os.path.normpath(get_path_opt_working(set_name) + '/' + C.folder_rend)
    return p


def clear_rend_leaf(set_name: str):
    clear_folder(get_path_rend_leaf(set_name))


def clear_rend_refs(set_name: str):
    get_path_rend_reference(C.imaging_type_refl, get_path_opt_working(set_name))
    get_path_rend_reference(C.imaging_type_tran, get_path_opt_working(set_name))


def get_path_opt_target_file(set_name: str):
    """Reference reflectance and transmittance."""
    p = os.path.normpath(get_path_opt_target(set_name) + '/' + C.file_opt_target + C.postfix_text_data_format)
    return p


def get_image_folder(target_type: str, imaging_type: str, base_path: str):
    """Returns a path to correct folder according to given target and imaging type.
    Basepath should be the work folder.
    """

    if target_type == C.target_type_leaf:
        return os.path.normpath(base_path + '/' + C.folder_rend)
    elif target_type == C.target_type_ref:
        return get_path_rend_reference(imaging_type, base_path)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_leaf}. Was {target_type}.")


def get_path_rend_reference(imaging_type: str, base_path: str) -> os.path:
    """Returns the path to reflectance or transmittance reference folder."""

    if imaging_type == C.imaging_type_refl:
        p = os.path.normpath(base_path + '/' + C.folder_rend_ref_refl)
    elif imaging_type == C.imaging_type_tran:
        p = os.path.normpath(base_path + '/' + C.folder_rend_ref_tran)
    else:
        raise Exception(f"Imaging type {imaging_type} not recognized. Use {C.imaging_type_refl} or {C.imaging_type_tran}.")
    return p


def generate_image_file_name(imaging_type: str, wl: float):
    image_name = f"{imaging_type}_wl{wl:.2f}{C.postfix_image_format}"
    return image_name


def get_image_file_path(target_type: str, imaging_type: str, wl: float, base_path: str):
    """Returns a full path to an image of given wavelength."""

    image_name = generate_image_file_name(imaging_type, wl)
    if target_type == C.target_type_leaf:
        return os.path.normpath(base_path + '/' + C.folder_rend + '/' + image_name)
    elif target_type == C.target_type_ref:
        return os.path.normpath(get_path_rend_reference(imaging_type, base_path) + '/' + image_name)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_leaf}. Was {target_type}.")


def search_by_wl(target_type: str, imaging_type: str, wl: float, base_path: str) -> os.path:
    """Search a folder for an image of given wavelength.

    A path to the image is returned.

    :raises FileNotFoundError if not found
    """

    def almost_equals(f1: float, f2: float, epsilon=0.01):
        res = abs(f1 - f2) <= epsilon
        return res

    folder = get_image_folder(target_type, imaging_type, base_path)
    for filename in os.listdir(folder):
        image_wl = utils.parse_wl_from_image_name(filename)
        if almost_equals(wl, image_wl):
            return get_image_file_path(target_type,imaging_type,wl, base_path)

    # Did not find
    raise FileNotFoundError(f"Could not find {wl} nm image from {folder}.")


def clear_folder(path):
    norm_path = os.path.normpath(path)
    if os.path.exists:
        list(map(os.unlink, (os.path.join(norm_path, f) for f in os.listdir(norm_path))))
    else:
        logging.warning(f"No files to delete in '{norm_path}'.")

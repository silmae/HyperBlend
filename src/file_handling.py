"""
All filepaths are handled here. Offers some basic operations, such as, creating
the default folder structure, clearing certain folders and searchin for a certain
wavelength subresults.
"""

import logging
import os

from src import constants as C
from src import utils


def create_opt_folder_structure(set_name: str):
    """Check that the folder structure for optimization is OK. Create if not."""
    print(f"Creating folder structure to set path {os.path.abspath(get_path_opt_set(set_name))}")
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
    """Path to optimization root folder 'project_root/optimization'. """
    p = os.path.abspath(C.path_project_root + '/' + C.folder_opt)
    return p


def get_path_opt_set(set_name: str):
    """Path to optimization folder of given set 'project_root/optimization/<set_name>'. """
    p = os.path.abspath(get_path_opt_root() + '/' + set_name)
    return p


def get_path_opt_target(set_name: str):
    """Path to optimization target folder (measurements) of given set 'project_root/optimization/<set_name>/target'. """
    p = os.path.abspath(get_path_opt_set(set_name) + '/' + C.folder_opt_target)
    return p


def get_path_opt_working(set_name: str):
    """Path to optimization working folder (for renders) of given set 'project_root/optimization/<set_name>/working_temp'. """
    p = os.path.abspath(get_path_opt_set(set_name) + '/' + C.folder_opt_work)
    return p


def get_path_opt_result(set_name: str):
    """Path to optimization result folder of given set 'project_root/optimization/<set_name>/result'. """
    p = os.path.abspath(get_path_opt_set(set_name) + '/' + C.folder_opt_result)
    return p


def get_path_opt_result_plot(set_name: str):
    """Path to result plot folder of given set 'project_root/optimization/<set_name>/result/plot'.

    This folder contains the final result plot and all subresult plots.
    """
    p = os.path.abspath(get_path_opt_result(set_name) + '/' + C.folder_opt_plot)
    return p


def get_path_opt_subresult(set_name: str):
    """Path to subresult folder of given set 'project_root/optimization/<set_name>/result/subresults'.

    The folder contains all subresults per wavelength as toml formatted txt files.
    """
    p = os.path.abspath(get_path_opt_result(set_name) + '/' + C.folder_opt_subresult)
    return p


def subresult_exists(set_name: str, wl:float):
    """Tells whether a certain subresult exists within given set.

    :param set_name:
        Set name.
    :param wl:
        Wavelength to be found.
    :return:
        True, if the subresult file was found, False otherwise.
    """
    # FIXME this is the same that is used by tom_handler when creating the file. Collect to same place.
    p = get_path_opt_subresult(set_name) + '/' + f"subres_wl_{wl:.2f}" + C.postfix_text_data_format
    if os.path.exists(p):
        return True
    else:
        return False


def target_exists(set_name: str) -> bool:
    """Tells wether target for given set exits.

    :param set_name:
        Set name.
    :return:
        True, if target file was found, False otherwise.
    """
    return os.path.exists(get_path_opt_set(set_name))


def get_path_rend_leaf(set_name: str):
    """Path to leaf render folder of given set 'project_root/optimization/<set_name>/working_temp/rend'. """
    p = os.path.abspath(get_path_opt_working(set_name) + '/' + C.folder_rend)
    return p


def clear_rend_leaf(set_name: str):
    """Clears leaf render folder of given set 'project_root/optimization/<set_name>/working_temp/rend'. """
    clear_folder(get_path_rend_leaf(set_name))


def clear_rend_refs(set_name: str):
    """Clears reference render folders of given set 'project_root/optimization/<set_name>/working_temp/rend_refl_ref and rend_tran_ref'. """
    clear_folder(get_path_rend_reference(C.imaging_type_refl, get_path_opt_working(set_name)))
    clear_folder(get_path_rend_reference(C.imaging_type_tran, get_path_opt_working(set_name)))


def get_path_opt_target_file(set_name: str):
    """Path to optimization target file (measurements) of given set 'project_root/optimization/<set_name>/target/target.toml'. """
    p = os.path.abspath(get_path_opt_target(set_name) + '/' + C.file_opt_target + C.postfix_text_data_format)
    return p


def get_image_folder(target_type: str, imaging_type: str, base_path: str):
    """Returns a path to correct folder according to given target and imaging type.

    :param base_path:
        Path to  the working folder. Usually the one returned by get_path_opt_working() is correct and other paths
        should only be used for testing and debugging.
    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param target_type:
        String either 'leaf' or 'reference'. Use the ones listed in constants.py.
    """

    if target_type == C.target_type_leaf:
        return os.path.abspath(base_path + '/' + C.folder_rend)
    elif target_type == C.target_type_ref:
        return get_path_rend_reference(imaging_type, base_path)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_ref}. Was {target_type}.")


def get_path_rend_reference(imaging_type: str, base_path: str) -> os.path:
    """Returns the path to reflectance or transmittance reference folder.

    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param base_path:
        Path to  the working folder. Usually the one returned by get_path_opt_working() is correct and other paths
        should only be used for testing and debugging.
    """

    if imaging_type == C.imaging_type_refl:
        p = os.path.abspath(base_path + '/' + C.folder_rend_ref_refl)
    elif imaging_type == C.imaging_type_tran:
        p = os.path.abspath(base_path + '/' + C.folder_rend_ref_tran)
    else:
        raise Exception(f"Imaging type {imaging_type} not recognized. Use {C.imaging_type_refl} or {C.imaging_type_tran}.")
    return p


def generate_image_file_name(imaging_type: str, wl: float):
    """Generates a name for a rendered image based on given wavelength.

    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param wl:
        Wavelength.
    :return:
        Image name in the format that other parts of the code can understand.
    """
    image_name = f"{imaging_type}_wl{wl:.2f}{C.postfix_image_format}"
    return image_name


def get_image_file_path(target_type: str, imaging_type: str, wl: float, base_path: str):
    """Returns a full path to an image of given wavelength.

    :param target_type:
        String either 'leaf' or 'reference'. Use the ones listed in constants.py.
    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param wl:
        Wavelength.
    :param base_path:
        Path to  the working folder. Usually the one returned by get_path_opt_working() is correct and other paths
        should only be used for testing and debugging.
    :returns:
        Returns absolute path to the image.
    """

    image_name = generate_image_file_name(imaging_type, wl)
    if target_type == C.target_type_leaf:
        return os.path.abspath(base_path + '/' + C.folder_rend + '/' + image_name)
    elif target_type == C.target_type_ref:
        return os.path.abspath(get_path_rend_reference(imaging_type, base_path) + '/' + image_name)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_leaf}. Was {target_type}.")


def search_by_wl(target_type: str, imaging_type: str, wl: float, base_path: str) -> os.path:
    """Search a folder for an image of given wavelength.

    A path to the image is returned.

    :param target_type:
        String either 'leaf' or 'reference'. Use the ones listed in constants.py.
    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param wl:
        Wavelength.
    :param base_path:
        Path to the image folder. Usually the one returned by get_image_file_path() is correct and other paths
        should only be used for testing and debugging.
    :returns:
        Returns absolute path to the image.
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
    """Clears the folder in given path.

    :param path:
        Path to folder to be cleared.
    :return:
        None
    """
    norm_path = os.path.abspath(path)
    if os.path.exists:
        list(map(os.unlink, (os.path.join(norm_path, f) for f in os.listdir(norm_path))))
    else:
        logging.warning(f"No files to delete in '{norm_path}'.")

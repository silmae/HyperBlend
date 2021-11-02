"""
All filepaths are handled here. Offers some basic operations, such as, creating
the default folder structure, clearing certain folders and searchin for a certain
wavelength subresults.

Path getters are named like path_directory_xxx or path_file_xxx.

"""

import logging
import os

import src.data.file_names
from src import constants as C
from src.data import file_names as FN


def create_first_level_folders(set_name: str):
    """Create first level folders for a set.

    Should be called when a new optimization object is instanced.

    :param set_name:
        Set name.
    """

    if not os.path.exists(path_directory_optimization()):
        os.makedirs(path_directory_optimization())
    if not os.path.exists(path_directory_target(set_name)):
        os.makedirs(path_directory_target(set_name))
    if not os.path.exists(path_directory_sample_result(set_name)):
        os.makedirs(path_directory_sample_result(set_name))
    if not os.path.exists(path_directory_set_result(set_name)):
        os.makedirs(path_directory_set_result(set_name))


def create_opt_folder_structure_for_samples(set_name: str, sample_id: int):
    """Check that the folder structure for optimization is OK. Create if not.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id
    """

    logging.info(f"Creating optimization folder structure to  path {os.path.abspath(path_directory_set(set_name))}")

    sample_folder_name = f'{C.folder_sample_prefix}_{sample_id}'
    sample_path = join(path_directory_sample_result(set_name), sample_folder_name)

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    if not os.path.exists(path_directory_working(set_name, sample_id)):
        os.makedirs(path_directory_working(set_name, sample_id))
    if not os.path.exists(path_directory_rend_leaf(set_name, sample_id)):
        os.makedirs(path_directory_rend_leaf(set_name, sample_id))
    if not os.path.exists(path_directory_rend_reference(C.imaging_type_refl, path_directory_working(set_name, sample_id))):
        os.makedirs(path_directory_rend_reference(C.imaging_type_refl, path_directory_working(set_name, sample_id)))
    if not os.path.exists(path_directory_rend_reference(C.imaging_type_tran, path_directory_working(set_name, sample_id))):
        os.makedirs(path_directory_rend_reference(C.imaging_type_tran, path_directory_working(set_name, sample_id)))
    if not os.path.exists(path_directory_subresult(set_name, sample_id)):
        os.makedirs(path_directory_subresult(set_name, sample_id))


def list_target_ids(set_name: str):
    """Lists available targets by their id.

    Targets must be named 'target_X.toml' where X is a number that can be cast into int.

    :param set_name:
        Set name.
    :return:
        List of ids (int) that were found from target folder.
    """

    ids = []
    for filename in os.listdir(path_directory_target(set_name)):
        ids.append(FN.parse_sample_id(filename))
    return ids


def list_finished_sample_ids(set_name: str) -> str:
    """Lists samples that have been optimized in given set.

    :param set_name:
        Set name.
    :return:
        List of sample ids (int) that have an existing result in sample results folder.
    """

    ids = []
    for sample_folder_name in os.listdir(path_directory_sample_result(set_name)):
        p = join(path_directory_sample_result(set_name), sample_folder_name)
        for filename in os.listdir(p):
            if filename.startswith(C.file_sample_result) and filename.endswith(C.postfix_text_data_format):
                ids.append(FN.parse_sample_id(filename))
    return ids


def subresult_exists(set_name: str, wl: float, sample_id: int) -> bool:
    """Tells whether a certain subresult exists within given set and sample.

    This is used to skip optimization of wavelengths that already have a result.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id.
    :param wl:
        Wavelength to be found. Has to be accurate to 2 decimals to be found.
    :return:
        True, if the subresult file was found, False otherwise.
    """

    p = path_file_subresult(set_name, wl, sample_id)
    res = os.path.exists(p)
    return res


##########################################################################
# Paths to directories
##########################################################################


def path_directory_set_result(set_name: str) -> str:
    """Path to where set result is saved."""

    p = os.path.abspath(path_directory_set(set_name) + '/' + C.folder_set_result)
    return p


def path_directory_sample(set_name: str, sample_id: int) -> str:
    p = os.path.abspath(path_directory_sample_result(set_name) + '/' + f'{C.folder_sample_prefix}_{sample_id}')
    return p


def path_directory_sample_result(set_name: str) -> str:
    """Path to where sample results are saved."""

    p = os.path.abspath(path_directory_set(set_name) + '/' + C.folder_opt_sample_results)
    return p


def path_directory_optimization() -> str:
    """Path to top level optimization root folder.

    'project_root/optimization'
    """

    p = os.path.abspath(C.path_project_root + '/' + C.folder_opt)
    return p


def path_directory_set(set_name: str) -> str:
    """Path to set's folder.

    'project_root/optimization/<set_name>'
    """

    p = os.path.abspath(path_directory_optimization() + '/' + set_name)
    return p


def path_directory_target(set_name: str) -> str:
    """Path to target folder (measurements) of given set.

    'project_root/optimization/<set_name>/target'
    """

    p = os.path.abspath(path_directory_set(set_name) + '/' + C.folder_opt_sample_targets)
    return p


def path_directory_working(set_name: str, sample_id: int) -> str:
    """Path to top level working folder where rendering sub-folders reside.

    'project_root/optimization/<set_name>/working_temp'
    """

    p = os.path.abspath(path_directory_sample(set_name, sample_id) + '/' + C.folder_opt_work)
    return p


def path_directory_subresult(set_name: str, sample_id: int) -> str:
    """Path to subresult folder of given set 'project_root/optimization/<set_name>/result/subresults'.

    The folder contains all subresults per wavelength as toml formatted txt files.
    """

    p = os.path.abspath(path_directory_sample(set_name, sample_id) + '/' + C.folder_opt_subresult)
    return p


def path_directory_rend_leaf(set_name: str, sample_id: int) -> str:
    """Path to leaf render folder of given set.

    'project_root/optimization/<set_name>/working_temp/rend'.
    """

    p = os.path.abspath(path_directory_working(set_name, sample_id) + '/' + C.folder_rend)
    return p


def path_directory_render(rendering_target: str, imaging_type: str, base_path: str) -> str:
    """Returns a path to correct folder according to given target and imaging type.

    :param base_path:
        Path to  the working folder. Usually the one returned by path_directory_working() is correct and other paths
        should only be used for testing and debugging.
    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param rendering_target:
        String either 'leaf' or 'reference'. Use the ones listed in constants.py.
    """

    if rendering_target == C.target_type_leaf:
        return join(base_path, C.folder_rend)
    elif rendering_target == C.target_type_ref:
        return path_directory_rend_reference(imaging_type, base_path)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_ref}. Was {rendering_target}.")


def path_directory_rend_reference(imaging_type: str, base_path: str) -> str:
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


##########################################################################
# Paths to files
##########################################################################


def path_file_subresult(set_name: str, wl: float, sample_id: int) -> str:
    """Path to subresult file of given sample and wavelength."""

    p = join(path_directory_subresult(set_name, sample_id), FN.filename_subresult(wl))
    return p


def path_file_target(set_name: str, sample_id: int):
    """Path to optimization target file (measurements) of given set and sample. """

    p = join(path_directory_target(set_name), FN.filename_target(sample_id))
    return p


def path_file_default_starting_guess():
    """Path to the default starting guess to be used in optimization."""

    p = join(C.path_project_root, FN.filename_starting_guess())
    return p


def path_file_rendered_image(target_type: str, imaging_type: str, wl: float, base_path: str) -> str:
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

    image_name = FN.filename_rendered_image(imaging_type, wl)
    if target_type == C.target_type_leaf:
        return join(base_path, C.folder_rend, image_name)
    elif target_type == C.target_type_ref:
        return join(path_directory_rend_reference(imaging_type, base_path), image_name)
    else:
        raise Exception(f"Target type must be either {C.target_type_leaf} or {C.target_type_leaf}. Was {target_type}.")


##########################################################################
# Clearing directories
##########################################################################


def clear_all_rendered_images(set_name: str) -> None:
    """Clear all rendered images of finished samples. """

    ids = list_finished_sample_ids(set_name)
    for _, sample_id in enumerate(ids):
        clear_rend_leaf(set_name, sample_id)
        clear_rend_refs(set_name, sample_id)


def clear_rend_leaf(set_name: str, sample_id: int) -> None:
    """Clears leaf render folder of given set, but leave reference renders untouched. """

    clear_folder(path_directory_rend_leaf(set_name, sample_id))


def clear_rend_refs(set_name: str, sample_id: int) -> None:
    """Clears reference render folders of given set but leave leaf renders untouched. """

    clear_folder(path_directory_rend_reference(C.imaging_type_refl, path_directory_working(set_name, sample_id)))
    clear_folder(path_directory_rend_reference(C.imaging_type_tran, path_directory_working(set_name, sample_id)))


def clear_folder(path: str) -> None:
    """Clears the folder in given path."""

    norm_path = os.path.abspath(path)
    if os.path.exists(norm_path):
        list(map(os.unlink, (join(norm_path, f) for f in os.listdir(norm_path))))
    else:
        logging.warning(f"No files to delete in '{norm_path}'.")


##########################################################################
# Misc
##########################################################################


def search_by_wl(target_type: str, imaging_type: str, wl: float, base_path: str) -> str:
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
        """Custom float equality for our desired 2 decimal accuracy."""

        res = abs(f1 - f2) <= epsilon
        return res

    folder = path_directory_render(target_type, imaging_type, base_path)
    for filename in os.listdir(folder):
        image_wl = src.data.file_names.parse_wl_from_image_name(filename)
        if almost_equals(wl, image_wl):
            return path_file_rendered_image(target_type, imaging_type, wl, base_path)

    raise FileNotFoundError(f"Could not find {wl} nm image from {folder}.")


def join(*args) -> str:
    """Custom join function to avoid problems using os.path.join. """
    n = len(args)
    s = ''
    for i,arg in enumerate(args):
        if i == n-1:
            s = s + arg
        else:
            s = s + arg + '/'
    p = os.path.normpath(s)
    return p
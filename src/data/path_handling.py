"""
Paths to directories and files.
"""

import os

from src import constants as C
from src.data import file_names as FN


##########################################################################
# Paths to directories
##########################################################################


def path_directory_set_result(set_name: str) -> str:
    """Path to where set result is saved."""

    p = os.path.abspath(path_directory_set(set_name) + '/' + C.folder_set_result)
    return p


def path_directory_sample_result(set_name: str) -> str:
    """Path to where sample results are saved."""

    p = os.path.abspath(path_directory_set(set_name) + '/' + C.folder_opt_sample_results)
    return p


def path_directory_sample(set_name: str, sample_id: int) -> str:
    """Path to sample subfolder."""

    p = os.path.abspath(path_directory_sample_result(set_name) + '/' + f'{C.folder_sample_prefix}_{sample_id}')
    return p


def path_directory_leaf_measurement_sets() -> str:
    """Path to top level leaf measurement sets root folder.

    'project_root/leaf_measurement_sets'
    """

    p = os.path.abspath(C.path_project_root + '/' + C.folder_leaf_measurement_sets)
    return p


def path_directory_surface_model() -> str:
    """Path to leaf model directory
     where surface model parameters and neural networks is stored.

    If the directory does not exist, it is created.

    'project_root/leaf_model'
    """

    p = os.path.abspath(C.path_project_root + '/' + C.folder_leaf_model)

    if not os.path.exists(p):
        os.makedirs(p)

    return p


def path_directory_set(set_name: str) -> str:
    """Path to set's folder.

    'project_root/leaf_measurement_set/<set_name>'
    """

    p = os.path.abspath(path_directory_leaf_measurement_sets() + '/' + set_name)
    return p


def path_directory_target(set_name: str) -> str:
    """Path to target folder (measurements) of given set.

    'project_root/leaf_measurement_sets/<set_name>/sample_targets'
    """

    p = os.path.abspath(path_directory_set(set_name) + '/' + C.folder_opt_sample_targets)
    return p


def path_directory_working(set_name: str, sample_id: int) -> str:
    """Path to top level working folder where rendering sub-folders reside.

    'project_root/leaf_measurement_sets/<set_name>/working_temp'
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
        p = join(base_path, C.folder_rend_ref_refl)
    elif imaging_type == C.imaging_type_tran:
        p = join(base_path, C.folder_rend_ref_tran)
    else:
        raise Exception(f"Imaging type {imaging_type} not recognized. Use {C.imaging_type_refl} or {C.imaging_type_tran}.")
    return p


def path_directory_forest_scenes() -> str:
    """Top level scene directory."""

    p = join(C.path_project_root, 'scenes')
    return os.path.abspath(p)


def path_directory_sun_data() -> str:
    """Sun data directory."""

    p = join(C.path_project_root, 'sun_data')
    return p


def path_directory_sky_data() -> str:
    """Sky data directory."""

    p = join(C.path_project_root, 'sky_data')
    return p


def path_directory_blender_scripts() -> str:
    """Blender scripts directory."""

    p = join(C.path_project_root, 'src', 'blender_scripts')
    return p


def path_directory_forest_scene(forest_id: str) -> str:
    """Specific forest scene directory."""

    p = join(path_directory_forest_scenes(), f"scene_{forest_id}")
    return p


def path_directory_forest_rend(forest_id: str) -> str:
    """Forest rend directory."""

    p = join(path_directory_forest_scene(forest_id), 'rend')
    return p


def path_directory_forest_cube(forest_id: str) -> str:
    """Forest spectral cube directory."""

    p = join(path_directory_forest_scene(forest_id), 'cube')
    return p


def path_directory_forest_rend_spectral(forest_id: str) -> str:
    """Forest spectral rend directory."""

    p = join(path_directory_forest_rend(forest_id), 'spectral')
    return p


def path_directory_forest_rend_abundances(forest_id: str) -> str:
    """Forest abundance maps rend directory."""

    p = join(path_directory_forest_rend(forest_id), 'abundances')
    return p

##########################################################################
# Paths to files
##########################################################################


def path_file_surface_model_parameters() -> str:
    """Path to surface model parameter file."""

    p = join(path_directory_surface_model(), C.file_model_parameters + C.postfix_text_data_format)
    return p


def path_file_wl_result(set_name: str, wl: float, sample_id: int) -> str:
    """Path to wavelength result toml file of given sample and wavelength."""

    p = join(path_directory_subresult(set_name, sample_id), FN.filename_wl_result(wl))
    return p


def path_file_target(set_name: str, sample_id: int, resampled=False):
    """Path to leaf measurement set's target file (measurements) of given set and sample.

    :param resampled:
        If True, path to corresponding resampled file is returned instead. Default is False.
    """

    p = join(path_directory_target(set_name), FN.filename_target(sample_id, resampled=resampled))
    return p


def path_file_sampling(set_name: str):
    """Path to spectral resampling data of given set. """

    p = join(path_directory_target(set_name), C.file_sampling_data + C.postfix_text_data_format)
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


def path_forest_template():
    """Path to forest template blend file."""

    p = join(C.path_project_root, 'scene_forest_template.blend')
    return p


def path_file_forest_scene(forest_id: str) -> str:
    """Path to certain forest scene blend file."""

    p = join(path_directory_forest_scene(forest_id), FN.filename_forest_scene(forest_id))
    return p


def path_file_forest_reflectance_cube(forest_id: str) -> str:
    p = join(path_directory_forest_cube(forest_id), FN.filename_forest_reflectance_cube(forest_id))
    return p


def path_file_forest_reflectance_header(forest_id: str) -> str:
    p = join(path_directory_forest_cube(forest_id), FN.filename_forest_reflectance_header(forest_id))
    return p


def path_file_forest_leaf_csv(forest_id: str, leaf_index):
    p = join(path_directory_forest_scene(forest_id), FN.filename_leaf_material_csv(leaf_index))
    return p


def path_file_forest_sun_csv(forest_id: str):
    p = join(path_directory_forest_scene(forest_id), 'blender_sun.csv')
    return p


def path_file_forest_sky_csv(forest_id: str):
    p = join(path_directory_forest_scene(forest_id), 'blender_sky.csv')
    return p


def path_file_forest_rgb_csv(forest_id: str):
    p = join(path_directory_forest_scene(forest_id), 'rgb_colors.csv')
    return p


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

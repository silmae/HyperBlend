"""
All filepaths are handled here. Offers some basic operations, such as, creating
the default folder structure, clearing certain folders and searchin for a certain
wavelength subresults.

Path getters are named like path_directory_xxx or path_file_xxx.

"""

import logging
import os
import datetime
import shutil
import csv

from src import constants as C, plotter
from src.data import file_names as FN, toml_handling as TH, path_handling as PH


def copy_target(from_set, to_set):
    create_first_level_folders(set_name=to_set)

    sample_ids = list_target_ids(from_set)
    for sample_id in sample_ids:
        target = TH.read_target(set_name=from_set, sample_id=sample_id)
        TH.write_target(set_name=to_set, data=target, sample_id=sample_id)


def create_first_level_folders(set_name: str):
    """Create first level folders for a set.

    Should be called when a new optimization object is instanced.

    :param set_name:
        Set name.
    """

    if not os.path.exists(PH.path_directory_leaf_measurement_sets()):
        os.makedirs(PH.path_directory_leaf_measurement_sets())
    if not os.path.exists(PH.path_directory_target(set_name)):
        os.makedirs(PH.path_directory_target(set_name))
    if not os.path.exists(PH.path_directory_sample_result(set_name)):
        os.makedirs(PH.path_directory_sample_result(set_name))
    if not os.path.exists(PH.path_directory_set_result(set_name)):
        os.makedirs(PH.path_directory_set_result(set_name))


def create_opt_folder_structure_for_samples(set_name: str, sample_id: int):
    """Check that the folder structure for optimization is OK. Create if not.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id
    """

    logging.info(f"Creating optimization folder structure to  path {os.path.abspath(PH.path_directory_set(set_name))}")

    sample_folder_name = f'{C.folder_sample_prefix}_{sample_id}'
    sample_path = PH.join(PH.path_directory_sample_result(set_name), sample_folder_name)

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    if not os.path.exists(PH.path_directory_working(set_name, sample_id)):
        os.makedirs(PH.path_directory_working(set_name, sample_id))
    if not os.path.exists(PH.path_directory_rend_leaf(set_name, sample_id)):
        os.makedirs(PH.path_directory_rend_leaf(set_name, sample_id))
    if not os.path.exists(PH.path_directory_rend_reference(C.imaging_type_refl, PH.path_directory_working(set_name, sample_id))):
        os.makedirs(PH.path_directory_rend_reference(C.imaging_type_refl, PH.path_directory_working(set_name, sample_id)))
    if not os.path.exists(PH.path_directory_rend_reference(C.imaging_type_tran, PH.path_directory_working(set_name, sample_id))):
        os.makedirs(PH.path_directory_rend_reference(C.imaging_type_tran, PH.path_directory_working(set_name, sample_id)))
    if not os.path.exists(PH.path_directory_subresult(set_name, sample_id)):
        os.makedirs(PH.path_directory_subresult(set_name, sample_id))


def list_target_ids(set_name: str):
    """Lists available targets by their id.

    Targets must be named 'target_X.toml' where X is a number that can be cast into int.

    :param set_name:
        Set name.
    :return:
        List of ids (int) that were found from target folder.
    """

    ids = []
    for filename in os.listdir(PH.path_directory_target(set_name)):
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
    for sample_folder_name in os.listdir(PH.path_directory_sample_result(set_name)):
        p = PH.join(PH.path_directory_sample_result(set_name), sample_folder_name)
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

    p = PH.path_file_wl_result(set_name, wl, sample_id)
    res = os.path.exists(p)
    return res


def clear_all_rendered_images(set_name: str) -> None:
    """Clear all rendered images of finished samples. """

    ids = list_finished_sample_ids(set_name)
    for _, sample_id in enumerate(ids):
        clear_rend_leaf(set_name, sample_id)
        clear_rend_refs(set_name, sample_id)


def clear_rend_leaf(set_name: str, sample_id: int) -> None:
    """Clears leaf render folder of given set, but leave reference renders untouched. """

    clear_folder(PH.path_directory_rend_leaf(set_name, sample_id))


def clear_rend_refs(set_name: str, sample_id: int) -> None:
    """Clears reference render folders of given set but leave leaf renders untouched. """

    clear_folder(PH.path_directory_rend_reference(C.imaging_type_refl, PH.path_directory_working(set_name, sample_id)))
    clear_folder(PH.path_directory_rend_reference(C.imaging_type_tran, PH.path_directory_working(set_name, sample_id)))


def clear_folder(path: str) -> None:
    """Clears the folder in given path."""

    norm_path = os.path.abspath(path)
    if os.path.exists(norm_path):
        list(map(os.unlink, (PH.join(norm_path, f) for f in os.listdir(norm_path))))
    else:
        logging.warning(f"No files to delete in '{norm_path}'.")


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

    folder = PH.path_directory_render(target_type, imaging_type, base_path)
    for filename in os.listdir(folder):
        image_wl = FN.parse_wl_from_filename(filename)
        if almost_equals(wl, image_wl):
            return PH.path_file_rendered_image(target_type, imaging_type, wl, base_path)

    raise FileNotFoundError(f"Could not find {wl} nm image from {folder}.")


def expand(set_name: str) -> None:
    """Generate back files removed by reduce().

    Does not do anything for sets modeled with 'surf' or 'nn' models.

    NOTE: Can not generate rendered images.
    """

    sample_ids = list_target_ids(set_name)

    for sample_id in sample_ids:
        TH.make_sample_result(set_name, sample_id)
        plotter.plot_sample_result(set_name, sample_id, dont_show=True, save_thumbnail=True)

    TH.write_set_result(set_name)
    plotter.replot_wl_results(set_name)
    plotter.plot_set_result(set_name, dont_show=True, save_thumbnail=True)
    plotter.plot_set_errors(set_name, dont_show=True, save_thumbnail=True)


def reduce(set_name: str) -> None:
    """Removes wavelength-wise optimization history plots and cleans temp working directories.

    Does not do anything for sets modeled with 'surf' or 'nn' models.

    Useful for reducing file size when sharing over internet, for example.
    Use expand() method to generate files as they were.

    Reduced size is about 1/10 of original size.

    NOTE: rendered images can not be generated back after they are deleted.
    """

    clear_all_rendered_images(set_name)

    sample_ids = list_finished_sample_ids(set_name)
    logging.info(f"Removing generated plots from set '{set_name}'.")
    for sample_id in sample_ids:
        p = PH.path_directory_subresult(set_name, sample_id)
        file_list = os.listdir(p)
        if len(file_list) == 0:
            logging.info(f"Nothing to remove. Directory '{p}' already empty.")
            continue
        else:
            logging.info(f"Cleaning subresult '{sample_id}'.")
        for plot in file_list:
            plot_path = PH.join(p, plot)
            if plot_path.endswith(C.postfix_plot_image_format):
                os.unlink(plot_path)
                # print(plot_path)

def duplicate_scene_from_template():
    """Creates a uniquely named copy of a scene and returns its id."""

    now = datetime.datetime.now()
    scene_id = f"{now.day:02}{now.month:02}{now.year-2000}{now.hour:02}{now.minute:02}"
    # Use this for debugging
    # scene_id = "0123456789"

    # p_src = f"../scene_forest_template.blend"
    # scene_folder = f"../scenes/scene_{scene_id}"
    # rend_folder = scene_folder + '/rend'
    # animation_folder = rend_folder + '/spectral'
    if os.path.exists(PH.path_forest_template()):
        if not os.path.exists(PH.path_directory_forest_scene(scene_id)):
            os.makedirs(PH.path_directory_forest_scene(scene_id))
        # p_dest = f"../scenes/scene_{scene_id}/scene_forest_{scene_id}.blend"
        shutil.copy2(PH.path_forest_template(), PH.path_file_forest_scene(scene_id))
        if not os.path.exists(PH.path_directory_forest_rend(scene_id)):
            os.makedirs(PH.path_directory_forest_rend(scene_id))
        if not os.path.exists(PH.path_directory_forest_rend_spectral(scene_id)):
            os.makedirs(PH.path_directory_forest_rend_spectral(scene_id))
    else:
        raise RuntimeError(f"Forest template scene not found for duplication.")

    return scene_id


def copy_leaf_material_parameters_from(source_set_name, forest_id, leaf_id):
    """Reads spectral set result and copies it as a leaf material parameter file
    to be consumed by forest setup.

    Leaf material parameters are written as a csv file to give to specified forest scene.

    :param source_set_name:
        Source set name that must be found from HyperBlend/optimization directory.
    :param forest_id:
        Forest id to be set the leaf material parameters to.
    :param leaf_id:
        Leaf index (1,2,3) to set the material parameters to.
    """

    res = TH.read_set_result(source_set_name)
    wls = res[C.key_set_result_wls]
    ad = res[C.key_set_result_wl_ad_mean]
    sd = res[C.key_set_result_wl_sd_mean]
    ai = res[C.key_set_result_wl_ai_mean]
    mf = res[C.key_set_result_wl_mf_mean]

    with open(PH.path_file_forest_leaf_csv(forest_id, leaf_id), 'w+', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=' ', )

        row = ["band", "wavelength", "absorption_density", "scattering_density", "scattering_anisotropy", "mix_factor"]
        #just headers for human readibility

        writer.writerow(row)
        for i, wl in enumerate(wls):
            row = [i + 1, wl, ad[i], sd[i], ai[i], mf[i]]
            writer.writerow(row)

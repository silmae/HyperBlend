"""
TODO docs

"""

import logging
import os
import datetime
import shutil
import csv
import re  # regular expressions

from src import plotter, constants as C
from src.data import file_names as FN, toml_handling as TH, path_handling as PH

CSV_NEWLINE = ""
CSV_DELIMITER = " "


def copy_slab_simulation_target(from_set: str, to_set: str):
    """Copy slab simulation targets and sampling data as a new slab simulation.

    See also :term:`Slab simulation`.

    :param from_set:
        Set name of the measurement set to copy from.
    :param to_set:
        Set name of the measurement set to copy to.
    """

    # Initialize new set with proper directories
    create_top_level_slab_sim_directories(slab_simu_name=to_set)

    # Copy all targets and resampled targets if they exist
    sample_ids = list_target_ids(from_set)
    for sample_id in sample_ids:

        path_src_target = PH.file_slab_target(
            slab_sim_name=from_set, signal_id=sample_id, resampled=False
        )
        path_dst_target = PH.file_slab_target(
            slab_sim_name=to_set, signal_id=sample_id, resampled=False
        )
        if os.path.exists(path_src_target):
            shutil.copy2(path_src_target, path_dst_target)

        path_src_target_resampled = PH.file_slab_target(
            slab_sim_name=from_set, signal_id=sample_id, resampled=True
        )
        path_dst_target_resampled = PH.file_slab_target(
            slab_sim_name=to_set, signal_id=sample_id, resampled=True
        )
        if os.path.exists(path_src_target_resampled):
            shutil.copy2(path_src_target_resampled, path_dst_target_resampled)

    # Copy sampling
    src_sampling = PH.file_spectral_sampling(from_set)
    if os.path.exists(src_sampling):
        dst_sampling = PH.file_spectral_sampling(to_set)
        shutil.copy2(src_sampling, dst_sampling)


def create_top_level_slab_sim_directories(slab_simu_name: str):
    """Create top level directories for slab simulation.

    Should be called when a new leaf measurement set is created.

    :param slab_simu_name: Name of the slab simulation.
    """

    if not os.path.exists(PH.directory_top_slab_simulation()):
        os.makedirs(PH.directory_top_slab_simulation())
    if not os.path.exists(PH.directory_top_target(slab_simu_name)):
        os.makedirs(PH.directory_top_target(slab_simu_name))
    if not os.path.exists(PH.directory_top_result_signal(slab_simu_name)):
        os.makedirs(PH.directory_top_result_signal(slab_simu_name))
    # if not os.path.exists(PH.path_directory_set_result(set_name)):
    #     os.makedirs(PH.path_directory_set_result(set_name))


def create_signal_optimization_directories(slab_sim_name: str, signal_id: int):
    """Create directories for slab simulation signal optimization."""

    sample_folder_name = f"{C.signal_directory_prefix}_{signal_id}"
    sample_path = PH.join(
        PH.directory_top_result_signal(slab_sim_name), sample_folder_name
    )

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    if not os.path.exists(
        PH.directory_slab_optimization_working(slab_sim_name, signal_id)
    ):
        os.makedirs(PH.directory_slab_optimization_working(slab_sim_name, signal_id))
    if not os.path.exists(PH.directory_slab_working_rend(slab_sim_name, signal_id)):
        os.makedirs(PH.directory_slab_working_rend(slab_sim_name, signal_id))
    if not os.path.exists(
        PH.directory_slab_rend_reference(
            C.imaging_type_refl,
            PH.directory_slab_optimization_working(slab_sim_name, signal_id),
        )
    ):
        os.makedirs(
            PH.directory_slab_rend_reference(
                C.imaging_type_refl,
                PH.directory_slab_optimization_working(slab_sim_name, signal_id),
            )
        )
    if not os.path.exists(
        PH.directory_slab_rend_reference(
            C.imaging_type_tran,
            PH.directory_slab_optimization_working(slab_sim_name, signal_id),
        )
    ):
        os.makedirs(
            PH.directory_slab_rend_reference(
                C.imaging_type_tran,
                PH.directory_slab_optimization_working(slab_sim_name, signal_id),
            )
        )
    if not os.path.exists(PH.directory_optimization_result(slab_sim_name, signal_id)):
        os.makedirs(PH.directory_optimization_result(slab_sim_name, signal_id))


def list_target_ids(set_name: str):
    """Lists available leaf measurement targets by their id.

    Targets must be named 'target_X.toml' where X is a number that can be cast into int.

    :param set_name:
        Set name.
    :return:
        List of ids (int) that were found from target folder.
    """

    ids = []
    for filename in os.listdir(PH.directory_top_target(set_name)):
        if re.match(r"target_[0-9]+\.toml", filename):
            ids.append(FN.parse_sample_id(filename))
    return ids


def list_finished_sample_ids(set_name: str):
    """Lists leaf measurement samples that have their renderable leaf material parameters solved.

    :param set_name:
        Name of the leaf measurement set.
    :return:
        List of sample ids (int) that have an existing result in sample results folder.
    """

    ids = []
    for sample_folder_name in os.listdir(PH.directory_top_result_signal(set_name)):
        p = PH.join(PH.directory_top_result_signal(set_name), sample_folder_name)
        for filename in os.listdir(p):
            if filename.startswith(C.filename_result_signal) and filename.endswith(
                C.postfix_text_data_format
            ):
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

    p = PH.file_wl_result(set_name, sample_id, wl)
    res = os.path.exists(p)
    return res


def clear_all_rendered_images(set_name: str) -> None:
    """Clear all rendered images of finished samples."""

    ids = list_finished_sample_ids(set_name)
    for _, sample_id in enumerate(ids):
        clear_rend_leaf(set_name, sample_id)
        clear_rend_refs(set_name, sample_id)


def clear_rend_leaf(set_name: str, sample_id: int) -> None:
    """Clears leaf render folder of given set, but leave reference renders untouched."""

    clear_folder(PH.directory_slab_working_rend(set_name, sample_id))


def clear_rend_refs(set_name: str, sample_id: int) -> None:
    """Clears reference render folders of given set but leave leaf renders untouched."""

    clear_folder(
        PH.directory_slab_rend_reference(
            C.imaging_type_refl,
            PH.directory_slab_optimization_working(set_name, sample_id),
        )
    )
    clear_folder(
        PH.directory_slab_rend_reference(
            C.imaging_type_tran,
            PH.directory_slab_optimization_working(set_name, sample_id),
        )
    )


def clear_folder(path: str) -> None:
    """Clears the folder in given path."""

    norm_path = os.path.abspath(path)
    if os.path.exists(norm_path):
        list(map(os.unlink, (PH.join(norm_path, f) for f in os.listdir(norm_path))))
    else:
        logging.warning(f"No files to delete in '{norm_path}'.")


def expand(set_name: str) -> None:
    """Generate back files removed by reduce().

    Does not do anything for sets modeled with 'surf' or 'nn' models.

    NOTE: Can not generate rendered images.
    """

    sample_ids = list_target_ids(set_name)

    for sample_id in sample_ids:
        TH.make_sample_result(set_name, sample_id)
        plotter.plot_sample_result(
            set_name, sample_id, dont_show=True, save_thumbnail=True
        )

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
        p = PH.directory_optimization_result(set_name, sample_id)
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


def duplicate_system_simulation_scene(
    system_sim_to_duplicate: str = None, new_system_sim_name: str = None
) -> str:
    """Creates a duplicate of a system simulation Blender scene.

    Creates the necessary directory structure.

    :param system_sim_to_duplicate: If provided, a system_simulation scene with this
        name is duplicated. If not provided, the default template system_simulation is
        used. See also :term:`system_sim_name`.
    :param new_system_sim_name: If given, this will be the name of the new system
        simulation scene. If not provided, a name based on date time will be generated.

    :return: If new_system_sim_name is not provided, a name based on date time will
        be returned.

    :raises FileNotFoundError: If the system simulation scene to duplicate does not exist.
    """

    if new_system_sim_name is not None:
        dst_system_sim_name = new_system_sim_name
    else:
        now = datetime.datetime.now()
        dst_system_sim_name = (
            f"{now.day:02}{now.month:02}{now.year - 2000}{now.hour:02}{now.minute:02}"
        )

    if system_sim_to_duplicate is not None:
        source_path = PH.file_blend_system_simulation(system_sim_to_duplicate)
    else:
        source_path = PH.file_blend_system_simulation_template()

    if os.path.exists(source_path):
        if not os.path.exists(PH.directory_system_simulation(dst_system_sim_name)):
            os.makedirs(PH.directory_system_simulation(dst_system_sim_name))

        shutil.copy2(source_path, PH.file_blend_system_simulation(dst_system_sim_name))

        if not os.path.exists(PH.directory_system_sim_rend(dst_system_sim_name)):
            os.makedirs(PH.directory_system_sim_rend(dst_system_sim_name))
        if not os.path.exists(PH.directory_system_rend_spectral(dst_system_sim_name)):
            os.makedirs(PH.directory_system_rend_spectral(dst_system_sim_name))
        if not os.path.exists(
            PH.directory_system_rend_visibility_maps(dst_system_sim_name)
        ):
            os.makedirs(PH.directory_system_rend_visibility_maps(dst_system_sim_name))
    else:
        raise FileNotFoundError(
            f"System simulation scene not found for duplication from '{source_path}'. "
        )
    logging.info(
        f"System simulation scene copied with id '{dst_system_sim_name}' to "
        f"'{PH.directory_system_simulation(dst_system_sim_name)}'."
    )

    return dst_system_sim_name


def copy_leaf_material_parameters(
    forest_id: str, leaf_id: str, source_set_name: str, sample_id: int = None
):
    """Reads spectral leaf simulation result and copies it as a leaf material parameter file
    to be consumed by system_simulation setup.

    Leaf material parameters are written as a csv file to give to specified system_simulation scene.
    We use csv file instead of toml files because importing external packages, such as toml,
    into Blender's own Python environment is bit of a hassle. Csv files work just as well and
    they can be read with tools already included by default.

    :param forest_id:
        Forest id to be set the leaf material parameters to.
    :param source_set_name:
        Source set name that must be found from HyperBlend/leaf_measurement_sets/ directory.
    :param leaf_id:
        An id to be assigned to the leaf for later referencing. This will be put into the
        name of the file written.
    :param sample_id:
        Sample id (int) of the leaf measurement set. If `None`, set's average values will be used instead
        of a specific sample.
    """

    if sample_id is None:
        result_dict = TH.read_set_result(source_set_name)
        wls = result_dict[C.key_set_result_wls]
        ad = result_dict[C.key_set_result_wl_ad_mean]
        sd = result_dict[C.key_set_result_wl_sd_mean]
        ai = result_dict[C.key_set_result_wl_ai_mean]
        mf = result_dict[C.key_set_result_wl_mf_mean]

        plot_path = PH.file_slab_sim_result_plot(slab_sim_name=source_set_name)

    else:
        result_dict = TH.read_sample_result(
            set_name=source_set_name, sample_id=sample_id
        )
        wls = result_dict[C.key_sample_result_wls]
        ad = result_dict[C.key_sample_result_ad]
        sd = result_dict[C.key_sample_result_sd]
        ai = result_dict[C.key_sample_result_ai]
        mf = result_dict[C.key_sample_result_mf]

        folder = PH.directory_top_target(slab_sim_name=source_set_name)
        image_name = FN.filename_resample_plot(sample_id=sample_id)
        plot_path = PH.join(folder, image_name)

    # Copy leaf plot to scene dir for convenience
    folder = PH.directory_system_simulation(system_sim_name=forest_id)
    image_name = f"leaf_spectrum_plot{leaf_id}{C.postfix_plot_image_format}"
    dst_plot_path = PH.join(folder, image_name)
    try:
        shutil.copy2(plot_path, dst_plot_path)
    except FileNotFoundError:
        logging.warning(
            f"Could not find resampled target plot for copying from '{plot_path}'."
        )

    with open(
        PH.file_system_slab_csv(forest_id, leaf_id), "w+", newline=CSV_NEWLINE
    ) as csvfile:

        writer = csv.writer(
            csvfile,
            delimiter=CSV_DELIMITER,
        )

        header = [
            "band",
            "wavelength",
            "absorption_density",
            "scattering_density",
            "scattering_anisotropy",
            "mix_factor",
        ]
        writer.writerow(header)

        for i, wl in enumerate(wls):
            row = [i + 1, wl, ad[i], sd[i], ai[i], mf[i]]
            writer.writerow(row)


def write_blender_light_spectra(forest_id: str, wls, irradiances, lighting_type="sun"):
    """Write light spectra to a csv file that can be read by Blender script.

    :param forest_id:
        Id of the system_simulation scene to write to.
    :param wls:
        List of wavelengths to be written.
    :param irradiances:
        List of sun irradiances to be written.
    :param lighting_type:
         String - either 'sun' or 'sky'.
    """

    if lighting_type == "sun":
        p = PH.file_system_sim_light_spectra_csv(forest_id)
    elif lighting_type == "sky":
        p = PH.file_forest_sky_csv(forest_id)
    else:
        raise ValueError(
            f"Wrong lighting type. Expected file type either 'sun' or 'sky', was '{lighting_type}'."
        )

    with open(p, "w+", newline=CSV_NEWLINE) as csvfile:

        writer = csv.writer(
            csvfile,
            delimiter=CSV_DELIMITER,
        )

        header = ["band", "wavelength", "irradiance"]
        writer.writerow(header)

        for i, wl in enumerate(wls):
            row = [i + 1, wl, irradiances[i]]
            writer.writerow(row)


def read_blender_light_spectra(forest_id: str, lighting_type="sun"):
    """Read light spectra csv from a Blender script.

    :param forest_id:
        Id of the system_simulation scene to read from.
    :param lighting_type:
         String either 'sun' or 'sky'.
    :return:
        bands, wls, irradiances - each is a list of floats.
    """

    if lighting_type == "sun":
        p = PH.file_system_sim_light_spectra_csv(forest_id)
    elif lighting_type == "sky":
        p = PH.file_forest_sky_csv(forest_id)
    else:
        raise ValueError(
            f"Wrong lighting type. Expected file type either 'sun' or 'sky', was '{lighting_type}'."
        )

    with open(p, "r", newline=CSV_NEWLINE) as csvfile:

        reader = csv.reader(
            csvfile, delimiter=CSV_DELIMITER, quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader, None)  # skip the headers

        bands = []
        wls = []
        irradiances = []
        for row in reader:
            bands.append(row[0])
            wls.append(row[1])
            irradiances.append(row[2])

        return bands, wls, irradiances


def write_blender_rgb_colors(forest_id: str, rgb_dict: dict):

    p = PH.file_system_sim_rgb_colors_csv(system_sim_name=forest_id)

    with open(p, "w+", newline=CSV_NEWLINE) as csvfile:

        writer = csv.writer(
            csvfile,
            delimiter=CSV_DELIMITER,
        )

        header = ["item", "r", "g", "b"]
        writer.writerow(header)

        for key, value in rgb_dict.items():
            row = [key, value[0], value[1], value[2]]
            writer.writerow(row)


def write_blender_soil(forest_id: str, wls, reflectances):
    """Write soil reflectance spectra to a csv file that can be read by Blender script.

    :param forest_id:
        Id of the system_simulation scene to write to.
    :param wls:
        List of wavelengths to be written.
    :param reflectances:
        List of sun irradiances to be written.
    """

    p = PH.file_forest_soil_csv(system_sim_name=forest_id)

    with open(p, "w+", newline=CSV_NEWLINE) as csvfile:

        writer = csv.writer(
            csvfile,
            delimiter=CSV_DELIMITER,
        )

        header = ["band", "wavelength", "reflectance"]
        writer.writerow(header)

        for i, wl in enumerate(wls):
            row = [i + 1, wl, reflectances[i]]
            writer.writerow(row)

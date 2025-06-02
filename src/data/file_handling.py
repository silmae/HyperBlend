"""
TODO docs


TODO: there are lots of methods that could be replaced if we could use a
    python version that includes a toml reader. It is available at least in
    Blender 4.4.
"""

import logging
import os
import datetime
import shutil
import csv
import re  # regular expressions
from typing import Tuple, List

from src import plotter, constants as C
from src.data import file_names as FN, toml_handling as TH, path_handling as PH

CSV_NEWLINE = ""
CSV_DELIMITER = " "


def copy_slab_simulation_target(src_slab_sim_name: str, dst_slab_sim_name: str) -> None:
    """Copy slab simulation targets and sampling data as a new slab simulation.

    See also :term:`Slab simulation`.

    :param src_slab_sim_name: The name of the slab simulation to copy from.
    :param dst_slab_sim_name: The name of the slab simulation to copy to.
    """

    # Initialize new set with proper directories
    create_top_level_slab_sim_directories(slab_sim_name=dst_slab_sim_name)

    # Copy all targets and resampled targets if they exist
    sample_ids = list_target_ids(src_slab_sim_name)
    for sample_id in sample_ids:

        path_src_target = PH.file_slab_target(
            slab_sim_name=src_slab_sim_name, signal_id=sample_id, resampled=False
        )
        path_dst_target = PH.file_slab_target(
            slab_sim_name=dst_slab_sim_name, signal_id=sample_id, resampled=False
        )
        if os.path.exists(path_src_target):
            shutil.copy2(path_src_target, path_dst_target)

        path_src_target_resampled = PH.file_slab_target(
            slab_sim_name=src_slab_sim_name, signal_id=sample_id, resampled=True
        )
        path_dst_target_resampled = PH.file_slab_target(
            slab_sim_name=dst_slab_sim_name, signal_id=sample_id, resampled=True
        )
        if os.path.exists(path_src_target_resampled):
            shutil.copy2(path_src_target_resampled, path_dst_target_resampled)

    # Copy sampling
    src_sampling = PH.file_spectral_sampling(src_slab_sim_name)
    if os.path.exists(src_sampling):
        dst_sampling = PH.file_spectral_sampling(dst_slab_sim_name)
        shutil.copy2(src_sampling, dst_sampling)
    else:
        logging.warning(
            f"File '{src_sampling}' not found. Sampling data not copied to '{dst_slab_sim_name}'."
            f"You should copy the files manually."
        )


def create_top_level_slab_sim_directories(slab_sim_name: str) -> None:
    """Create top level directories for slab simulation.

    The creation of the directories for a certain signal is covered
    by :func:`create_slab_sim_signal_directories()`.

    The creation of the directories needed by the optimizer
    are covered by :func:`create_signal_optimization_directories()`.

    These are separate methods because the others need also the signal id.

    :param slab_sim_name: Name of the slab simulation.
    """

    dirs_to_create = [
        PH.directory_top_slab_simulation(),
        PH.directory_top_target(slab_sim_name),
        PH.directory_top_result_signal(slab_sim_name),
    ]

    for path in dirs_to_create:
        if not os.path.exists(path):
            os.makedirs(path)

    logging.info(f"Top level slab simulation directories created.")
    printable_list = "\n".join(dirs_to_create)
    logging.debug(f"Created directories: {printable_list}")

    # if not os.path.exists(PH.directory_top_slab_simulation()):
    #     os.makedirs(PH.directory_top_slab_simulation())
    # if not os.path.exists(PH.directory_top_target(slab_sim_name)):
    #     os.makedirs(PH.directory_top_target(slab_sim_name))
    # if not os.path.exists(PH.directory_top_result_signal(slab_sim_name)):
    #     os.makedirs(PH.directory_top_result_signal(slab_sim_name))


def create_top_level_system_sim_directories(system_sim_name: str) -> None:
    """Create top level directories for system simulation."""

    dirs_to_create = [
        PH.directory_system_simulation(system_sim_name),
        PH.directory_system_sim_rend(system_sim_name),
        PH.directory_system_rend_spectral(system_sim_name),
        PH.directory_system_rend_visibility_maps(system_sim_name),
    ]

    for path in dirs_to_create:
        if not os.path.exists(path):
            os.makedirs(path)

    logging.info(f"Top level system simulation directories created.")
    printable_list = "\n".join(dirs_to_create)
    logging.debug(f"Created directories: {printable_list}")


def create_slab_sim_signal_directories(slab_sim_name: str, signal_id: int) -> None:
    """Create directories for slab simulation result signal.

    This is needed regardless of the used solver for storing the results.
    """

    dirs_to_create = [
        PH.directory_result_signal(slab_sim_name=slab_sim_name, signal_id=signal_id),
    ]

    for path in dirs_to_create:
        if not os.path.exists(path):
            os.makedirs(path)

    logging.info(f"Slab simulation's signal directories created.")
    printable_list = "\n".join(dirs_to_create)
    logging.debug(f"Created directories: {printable_list}")


def create_signal_optimization_directories(slab_sim_name: str, signal_id: int) -> None:
    """Create directories for slab simulation signal optimization solver."""

    p_working = PH.directory_slab_optimization_working(slab_sim_name, signal_id)
    dirs_to_create = [
        PH.directory_slab_working_rend(slab_sim_name, signal_id),
        PH.directory_slab_rend_reference(C.imaging_type_refl, p_working),
        PH.directory_slab_rend_reference(C.imaging_type_tran, p_working),
        PH.directory_optimization_result(slab_sim_name, signal_id),
    ]

    for path in dirs_to_create:
        if not os.path.exists(path):
            os.makedirs(path)

    logging.info(f"Slab simulation's signal directories for optimization created.")
    printable_list = "\n".join(dirs_to_create)
    logging.debug(f"Created directories: {printable_list}")

    # if not os.path.exists(p_working):
    #     os.makedirs(p_working)
    #
    # p = PH.directory_slab_working_rend(slab_sim_name, signal_id)
    # if not os.path.exists(p):
    #     os.makedirs(p)
    #
    # p = PH.directory_slab_rend_reference(C.imaging_type_refl, p_working)
    # if not os.path.exists(p):
    #     os.makedirs(p)
    #
    # p = PH.directory_slab_rend_reference(C.imaging_type_tran, p_working)
    # if not os.path.exists(p):
    #     os.makedirs(p)
    #
    # p = PH.directory_optimization_result(slab_sim_name, signal_id)
    # if not os.path.exists(p):
    #     os.makedirs(p)


def list_target_ids(slab_sim_name: str) -> list[int]:
    """Lists available signal targets by their id.

    Targets must be named 'target_X.toml' where X is a number that can be cast into int.

    :param slab_sim_name: Name of the slab simulation.

    :return: List of target ids (int) that were found from target folder.
    """

    ids = []
    for filename in os.listdir(PH.directory_top_target(slab_sim_name)):
        if re.match(r"target_[0-9]+\.toml", filename):
            ids.append(FN.parse_sample_id(filename))
    return ids


def list_finished_result_signal_ids(slab_sim_name: str) -> list[int]:
    """Lists signal ids that have an existing result in the slab simulation's results directory."""

    ids = []
    for signal_dir_name in os.listdir(PH.directory_top_result_signal(slab_sim_name)):
        p = PH.join(PH.directory_top_result_signal(slab_sim_name), signal_dir_name)
        for filename in os.listdir(p):
            if filename.startswith(C.filename_result_signal) and filename.endswith(
                C.postfix_text_data_format
            ):
                ids.append(FN.parse_sample_id(filename))
    return ids


def optimization_wl_result_exists(
    slab_sim_name: str, wl: float, signal_id: int
) -> bool:
    """Tells whether a result exists for a given wavelength within given signal and slab simulation.

    This is used to skip optimization of wavelengths that already have a result.

    :param slab_sim_name: Name of the slab simulation.
    :param signal_id: The id of the signal.
    :param wl: Wavelength to be searched for. Has to be accurate to 2 decimals to be found.

    :returns: True if the result exists, False otherwise.
    """

    p = PH.file_wl_result(slab_sim_name, signal_id, wl)
    res = os.path.exists(p)
    return res


def clear_all_rendered_images(slab_sim_name: str) -> None:
    """Clear all rendered images that have their optimization result ready."""

    ids = list_finished_result_signal_ids(slab_sim_name)
    for _, sample_id in enumerate(ids):
        clear_rend_slab(slab_sim_name, sample_id)
        clear_rend_refs(slab_sim_name, sample_id)


def clear_rend_slab(slab_sim_name: str, signal_id: int) -> None:
    """Clears slab render directory of the given simulation.

    Leaves the slab reference renders untouched.
    """

    clear_directory(PH.directory_slab_working_rend(slab_sim_name, signal_id))


def clear_rend_refs(slab_sim_name: str, signal_id: int) -> None:
    """Clears the slab **reference** renders but leaves the slab renders untouched."""

    clear_directory(
        PH.directory_slab_rend_reference(
            C.imaging_type_refl,
            PH.directory_slab_optimization_working(slab_sim_name, signal_id),
        )
    )
    clear_directory(
        PH.directory_slab_rend_reference(
            C.imaging_type_tran,
            PH.directory_slab_optimization_working(slab_sim_name, signal_id),
        )
    )


def clear_directory(path: str) -> None:
    """Clears all files from the directory in given path."""

    norm_path = os.path.abspath(path)
    if os.path.exists(norm_path):
        list(map(os.unlink, (PH.join(norm_path, f) for f in os.listdir(norm_path))))
    else:
        logging.warning(f"No files to delete in '{norm_path}'.")


def expand(slab_sim_name: str) -> None:
    """Re-generate files removed by :func:`reduce()`.

    Doesn't do anything for slab simulations solved with :mod:`slab_model.surf`
    or :mod:`slab_model.nn` models.

    .. note::
        Can not re-generate rendered images but will generate optimization history.
        This takes quite a bit of time, so only use if you really need it.
    """

    signal_ids = list_target_ids(slab_sim_name)

    for signal_id in signal_ids:
        TH.make_signal_result(slab_sim_name, signal_id)
        plotter.plot_signal_result(
            slab_sim_name, signal_id, dont_show=True, save_thumbnail=True
        )

    TH.write_slab_sim_result(slab_sim_name)
    plotter.replot_wl_results(slab_sim_name)
    plotter.plot_slab_sim_result(slab_sim_name, dont_show=True, save_thumbnail=True)
    plotter.plot_slab_sim_errors(slab_sim_name, dont_show=True, save_thumbnail=True)


def reduce(slab_sim_name: str) -> None:
    """Removes wavelength-wise optimization history plots and cleans up temp working directories.

    Doesn't do anything for slab simulations solved with :mod:`slab_model.surf`
    or :mod:`slab_model.nn` models.

    Useful for reducing file size when sharing over internet, for example.
    Use :func:`expand()` method to re-generate most of the files as they were.

    Reduced size is about 1/10 of original size.

    .. note::
        Rendered images can not be generated back after they are deleted.
    """

    clear_all_rendered_images(slab_sim_name)

    sample_ids = list_finished_result_signal_ids(slab_sim_name)
    logging.info(f"Removing generated plots from slab simulation '{slab_sim_name}'.")
    for sample_id in sample_ids:
        p = PH.directory_optimization_result(slab_sim_name, sample_id)
        file_list = os.listdir(p)
        if len(file_list) == 0:
            logging.info(f"Nothing to remove. Directory '{p}' already empty.")
            continue
        else:
            logging.info(f"Cleaning wavelength results of sample '{sample_id}'.")
        for plot in file_list:
            plot_path = PH.join(p, plot)
            if plot_path.endswith(C.postfix_plot_image_format):
                os.unlink(plot_path)


def duplicate_system_simulation_scene(
    src_system_sim_name: str = None, dst_system_sim_name: str = None
) -> str:
    """Creates a duplicate of a system simulation Blender scene.

    Creates the necessary directory structure.

    :param src_system_sim_name: If provided, a system_simulation scene with this
        name is duplicated. If not provided, the default template system simulation is
        used. See also :term:`system_sim_name`.
    :param dst_system_sim_name: If given, this will be the name of the new system
        simulation scene. If not provided, a name based on date time will be generated.

    :return: The name of the duplicated system simulation scene. If `dst_system_sim_name`
        was provided, the same name is returned. Otherwise, a generated name is returned.

    :raises FileNotFoundError: If the system simulation scene to duplicate does not exist.
    """

    if dst_system_sim_name is not None:
        dst_system_sim_name = dst_system_sim_name
    else:
        now = datetime.datetime.now()
        dst_system_sim_name = (
            f"{now.day:02}{now.month:02}{now.year - 2000}{now.hour:02}{now.minute:02}"
        )

    if src_system_sim_name is None:
        source_path = PH.file_blend_system_simulation_template(src_system_sim_name)
    else:
        source_path = PH.file_blend_system_simulation(src_system_sim_name)

    if os.path.exists(source_path):
        create_top_level_system_sim_directories(system_sim_name=dst_system_sim_name)
        shutil.copy2(source_path, PH.file_blend_system_simulation(dst_system_sim_name))
        logging.info(
            f"System simulation scene copied with id '{dst_system_sim_name}' to "
            f"'{PH.directory_system_simulation(dst_system_sim_name)}'."
        )
    else:
        raise FileNotFoundError(
            f"System simulation scene not found for duplication from '{source_path}'. "
        )

    return dst_system_sim_name


def copy_slab_material_parameters(
    system_sim_name: str,
    slab_material_name: str,
    src_slab_sim_name: str,
    signal_id: int = None,
):
    """Reads slab simulation result and copies it as a slab material parameter file
    to be consumed by :mod:`blender_scripts.bs_setup_forest`.

    Slab material parameters are written as a csv file to give to specified system_simulation scene.
    We use csv file instead of toml files because importing external packages, such as toml,
    into Blender's own Python environment is a bit of a hassle. Csv files work just as well and
    they can be read with tools already included by default.

    TODO: check how this is actually used. The material names should be generalized at some point

    .. note::
        This may change as new Python version with included toml is in use in HyperBlend.

    :param system_sim_name: Name of the system simulation.
    :param src_slab_sim_name: Name of the slab simulation to copy from.
    :param slab_material_name: Name of the slab material. This will be put into the
        name of the file written and is used by the scripts in :mod:`blender_scripts`.
    :param signal_id: Signal id of the slab simulation. If `None`, slab simulation's
        average values will be used instead of a specific signal.
    """

    if signal_id is None:
        result_dict = TH.read_set_result(src_slab_sim_name)
        wls = result_dict[C.key_set_result_wls]
        ad = result_dict[C.key_set_result_wl_ad_mean]
        sd = result_dict[C.key_set_result_wl_sd_mean]
        ai = result_dict[C.key_set_result_wl_ai_mean]
        mf = result_dict[C.key_set_result_wl_mf_mean]

        plot_path = PH.file_slab_sim_result_plot(slab_sim_name=src_slab_sim_name)

    else:
        result_dict = TH.read_sample_result(
            set_name=src_slab_sim_name, sample_id=signal_id
        )
        wls = result_dict[C.key_sample_result_wls]
        ad = result_dict[C.key_sample_result_ad]
        sd = result_dict[C.key_sample_result_sd]
        ai = result_dict[C.key_sample_result_ai]
        mf = result_dict[C.key_sample_result_mf]

        folder = PH.directory_top_target(slab_sim_name=src_slab_sim_name)
        image_name = FN.filename_resample_plot(sample_id=signal_id)
        plot_path = PH.join(folder, image_name)

    # Copy slab result plot to scene dir for convenience
    folder = PH.directory_system_simulation(system_sim_name=system_sim_name)

    # TODO: move to filenames and use consistently
    image_name = f"leaf_spectrum_plot{slab_material_name}{C.postfix_plot_image_format}"

    dst_plot_path = PH.join(folder, image_name)
    try:
        shutil.copy2(plot_path, dst_plot_path)
    except FileNotFoundError:
        logging.warning(
            f"Could not find resampled target plot for copying from '{plot_path}'."
        )

    p = PH.file_system_slab_csv(system_sim_name, slab_material_name)
    with open(p, "w+", newline=CSV_NEWLINE) as csvfile:

        writer = csv.writer(csvfile, delimiter=CSV_DELIMITER)

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


def write_blender_light_spectra(
    system_sim_name: str, wls: list[float], irradiances, lighting_type="sun"
):
    """Write light spectra to a csv file that can be read by Blender script.

    TODO: accept any general light source name. Sun and sky can be in a separate
        method if needed, but really, maybe not necessary.

    :param system_sim_name: Name of the system_simulation scene to write to.
    :param wls: List of wavelengths to be written.
    :param irradiances: List of light irradiances to be written.
    :param lighting_type: Either 'sun' or 'sky'.
    """

    if lighting_type == "sun":
        p = PH.file_system_sim_light_spectra_csv(
            system_sim_name=system_sim_name, light_file_name=C.file_blender_default_sun
        )
    elif lighting_type == "sky":
        p = PH.file_system_sim_light_spectra_csv(
            system_sim_name=system_sim_name, light_file_name=C.file_blender_default_sky
        )
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


# def read_blender_light_spectra(
#     system_sim_name: str, lighting_type="sun"
# ) -> tuple[list[float], list[float], list[float]]:
#     """Read light spectra csv from a Blender script.
#
#     :param system_sim_name: Name of the system simulation to read the light file from.
#     :param lighting_type: String either 'sun' or 'sky'.
#
#     :returns: bands, wls, irradiances - each is a list of floats.
#     """
#
#     if lighting_type == "sun":
#         p = PH.file_system_sim_light_spectra_csv(
#             system_sim_name=system_sim_name, light_file_name=C.file_blender_default_sun
#         )
#     elif lighting_type == "sky":
#         p = PH.file_system_sim_light_spectra_csv(
#             system_sim_name=system_sim_name, light_file_name=C.file_blender_default_sky
#         )
#     else:
#         raise ValueError(
#             f"Light type not recognized. Expected file type either 'sun' or 'sky', "
#             f"was '{lighting_type}'."
#         )
#
#     with open(p, "r", newline=CSV_NEWLINE) as csvfile:
#
#         reader = csv.reader(
#             csvfile, delimiter=CSV_DELIMITER, quoting=csv.QUOTE_NONNUMERIC
#         )
#         next(reader, None)  # skip the headers
#
#         bands = []
#         wls = []
#         irradiances = []
#         for row in reader:
#             bands.append(row[0])
#             wls.append(row[1])
#             irradiances.append(row[2])
#
#         return bands, wls, irradiances


def write_blender_rgb_colors(system_sim_name: str, rgb_dict: dict) -> None:
    """Write RGB colors to a csv file that can be read by :mod:`blender_scripts`.

    :param system_sim_name Name of the system simulation.
    :param rgb_dict: Dictionary of RGB colors to be written. The keys are the
        names of the colors
    """

    p = PH.file_system_sim_rgb_colors_csv(system_sim_name=system_sim_name)

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


def write_blender_soil(
    system_sim_name: str, wls: list[float], reflectances: list[float]
) -> None:
    """Write soil reflectance spectra to a csv file that can be read by Blender script.

    :param system_sim_name: Name of the system simulation.
    :param wls: List of wavelengths to be written.
    :param reflectances: List of reflectances to be written.
    """

    p = PH.file_forest_soil_csv(system_sim_name=system_sim_name)

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

"""
Paths to directories and files. Almost all methods return a single string except some
search methods that return a list of strings.

Methods to find directories that contain subdirectories have the word 'top' in their name.

Methods to find directories that contain program code of HyperBlend have the
word 'code' in their name.

"""

import logging
import os
from typing import List

import constants as C
from data import file_names as FN

from src import constants as C
from src.data import file_names as FN


##########################################################################
# Paths to directories
##########################################################################


# Top level directories


def directory_project_root() -> str:
    """Path to project root directory."""

    p = os.path.abspath(C.path_project_root)
    return p


def directory_log() -> str:
    """Path to logging directory."""

    p = join(C.path_project_root, C.dirname_log)
    return p


def directory_internal() -> str:
    """Path to directory containing HyperBlend internal files."""

    p = join(C.path_project_root, C.dirname_internal)
    return p


def directory_top_system_simulation() -> str:
    """Path to top level system simulation directory."""

    p = join(C.path_project_root, C.dirname_system_sim)
    return p


def directory_light_spectra() -> str:
    """Path to light spectra directory that contains all spectra for light sources."""

    p = join(C.path_project_root, C.dirname_light_spectra)
    return p


def directory_reflectance_spectra() -> str:
    """Path to reflectance spectra directory.

    For light spectra, use :func:`directory_light_spectra()`.

    For materials that are reflective and transmissive, use TODO
    """

    p = join(C.path_project_root, C.dirname_reflectance_spectra)
    return p


# Code directories


def directory_code_source() -> str:
    """Path to source code directory."""

    p = join(C.path_project_root, C.dirname_source)
    return p


def directory_code_blender_scripts() -> str:
    """Blender scripts directory."""

    p = join(directory_code_source(), C.dirname_blender_scripts)
    return p


def directory_code_definitions() -> str:
    """HyperBlend's internal definitions."""

    p = join(directory_code_source(), C.dirname_definitions)
    return p


def directory_code_soil() -> str:
    """Soil code directory that contain gsv spectra vectors used for gsv generation."""

    p = join(directory_code_source(), C.dirname_gsv)
    return p


# Simulation directories


def directory_slab_model(slab_model_name: str = None) -> str:
    """Path to an arbitrary slab model directory.

    It stores surface model parameters, a neural network, and a starting guess.

    :param slab_model_name: If None, path to the default slab model is returned.

    :return: Returns path to the slab model directory.
    """

    if slab_model_name is None:
        p = join(
            C.path_project_root, C.dirname_slab_models, C.dirname_slab_models_default
        )
    else:
        p = join(C.path_project_root, C.dirname_slab_models, slab_model_name)
    return p


def directory_top_slab_simulation() -> str:
    """Path to top level directory where all slab simulations are stored."""

    p = join(C.path_project_root, C.dirname_slab_simulation)
    return p


def directory_slab_simulation(slab_sim_name: str) -> str:
    """Path to a specific slab simulation.

    The top level directory is given by :func:`directory_top_slab_simulation()`.
    """

    p = join(directory_top_slab_simulation(), slab_sim_name)
    return p


def directory_top_result_signal(slab_sim_name: str) -> str:
    """Result signals of a certain slab simulation. See :term:`slab_sim_name`"""

    p = join(directory_slab_simulation(slab_sim_name), C.dirname_result_signal)
    return p


def directory_result_signal(slab_sim_name: str, signal_id: int) -> str:
    """Path to a directory where a specific signal resulted from slab simulation is saved."""

    # TODO the name should come from the name handler
    p = join(
        directory_top_result_signal(slab_sim_name),
        f"{C.signal_directory_prefix}_{signal_id}",
    )
    return p


def directory_top_target(slab_sim_name: str) -> str:
    """Path to the top level target signal directory."""

    p = join(directory_slab_simulation(slab_sim_name), C.dirname_opt_target_signal)
    return p


def directory_slab_optimization_working(slab_sim_name: str, signal_id: int) -> str:
    """Path to top level working folder of the slab simulation for given signal.

    Optimization renders images to define the virtual slab material parameters.
    Those images are stored in the working folder.
    """

    p = join(
        directory_result_signal(slab_sim_name, signal_id),
        C.dirname_slab_sim_working_temp,
    )
    return p


def directory_optimization_result(slab_sim_name: str, signal_id: int) -> str:
    """Path to optimization result directory of given slab simulation and signal id.

    Only used if slab model is run in optimization mode.
    """

    p = join(
        directory_result_signal(slab_sim_name, signal_id),
        C.dirname_optimization_results,
    )
    return p


def directory_slab_working_rend(slab_sim_name: str, signal_id: int) -> str:
    """Path to the rendering directory of given slab simulation and signal id."""

    p = join(
        directory_slab_optimization_working(slab_sim_name, signal_id),
        C.dirname_slab_sim_rend,
    )
    return p


def directory_system_simulation(system_sim_name: str) -> str:
    """Specific system simulation scene directory."""

    # TODO the name should come from the name handler
    p = join(directory_top_system_simulation(), f"scene_{system_sim_name}")
    return p


def directory_system_sim_rend(system_sim_name: str) -> str:
    """Rendering directory of the system simulation."""

    p = join(directory_system_simulation(system_sim_name), C.dirname_system_sim_rend)
    return p


def directory_system_spectral_cube(system_sim_name: str) -> str:
    """Spectral cube resulting from a system simulation is stored here."""

    p = join(directory_system_simulation(system_sim_name), C.dirname_system_cube)
    return p


def directory_system_rend_spectral(system_sim_name: str) -> str:
    """System simulation spectral rendering directory."""

    p = join(
        directory_system_sim_rend(system_sim_name), C.dirname_system_sim_spectral_rend
    )
    return p


def directory_system_rend_visibility_maps(system_sim_name: str) -> str:
    """System simulation render directory for visibility maps of materials."""

    p = join(
        directory_system_sim_rend(system_sim_name),
        C.dirname_system_sim_visibility_maps_rend,
    )
    return p


##########################################################################
# Paths to files
##########################################################################


def file_surface_model_parameters(slab_model_name: str = None) -> str:
    """Path to surface model parameter file. See :term:`slab_model_name`.

    :param slab_model_name: If None, path to the default slab model is returned.
    """

    if slab_model_name is None:
        p = join(directory_slab_model(), C.slab_surf_name)
    else:
        p = join(
            directory_slab_model(slab_model_name=slab_model_name), C.slab_surf_name
        )

    return p


def file_wl_result(slab_sim_name: str, signal_id: int, wl: float) -> str:
    """Path to wavelength result toml file of given slab simulation, signal id, and wavelength."""

    p = join(
        directory_optimization_result(slab_sim_name, signal_id),
        FN.filename_wl_result(wl),
    )
    return p


def file_signal_result(slab_sim_name: str, signal_id: int) -> str:
    """Path to signal result toml file of given slab simulation and signal id."""

    p = join(
        directory_result_signal(slab_sim_name, signal_id),
        FN.filename_sample_result(signal_id),
    )
    return p


def file_signal_result_plot(slab_sim_name: str, signal_id: int) -> str:
    """Path to signal result plot of given slab simulation."""

    p = join(
        directory_result_signal(slab_sim_name, signal_id),
        FN.filename_signal_result_plot(signal_id),
    )
    return p


def file_slab_sim_result(slab_sim_name: str) -> str:
    """Path to slab simulation result toml file of given slab simulation."""

    filename = C.filename_slab_sim_result + C.postfix_text_data_format
    p = join(directory_slab_simulation(slab_sim_name=slab_sim_name), filename)
    return p


def file_slab_sim_result_plot(slab_sim_name: str) -> str:
    """Path to slab simulation result plot file of given slab simulation."""

    filename = C.filename_slab_sim_result + C.postfix_plot_image_format
    p = join(directory_slab_simulation(slab_sim_name=slab_sim_name), filename)
    return p


def file_slab_sim_error_plot(slab_sim_name: str) -> str:
    """Path to slab simulation error plot of given slab simulation."""

    filename = C.filename_slab_sim_error_plot + C.postfix_plot_image_format
    p = join(directory_slab_simulation(slab_sim_name=slab_sim_name), filename)
    return p


def file_slab_target(slab_sim_name: str, signal_id: int, resampled=False) -> str:
    """Path to slab simulation target spectrum file of given slab simulation and signal.

    :param slab_sim_name: Slab simulation name.
    :param signal_id: Signal id.
    :param resampled: If True, path to corresponding resampled file is returned instead.
        Default is False.
    """

    p = join(
        directory_top_target(slab_sim_name=slab_sim_name),
        FN.filename_target_signal(signal_id=signal_id, resampled=resampled),
    )
    return p


def file_spectral_sampling(slab_sim_name: str) -> str:
    """Path to spectral resampling data of given slab simulation."""

    p = join(
        directory_top_target(slab_sim_name),
        C.file_sampling_data + C.postfix_text_data_format,
    )
    return p


def file_starting_guess(slab_model_name=None) -> str:
    """Path to the starting guess parameter file to be used in optimization.

    :param slab_model_name: If None given, the default slab model's starting guess is used.
    """

    # TODO move the filename to constants
    filename = "starting_guess" + C.postfix_text_data_format
    p = join(directory_slab_model(slab_model_name=slab_model_name), filename)
    return p


def file_blend_system_simulation_template(
    template_name: str = None,
) -> str:
    """Path to system simulation template Blender file found in directory 'Internal/'.

    The .blend extension is added automatically if not given.

    TODO: this doesn't make any sense anymore as there can be only one template...
        no! There can be other templates. But currently there is only one. So the
        caller must take care if they actually want a template or some other
        scene file.

    :param template_name: Name of the template file. If None given,
        the default template is used.
    """

    if template_name is None:
        template_name = C.filename_system_sim_forest_template

    if not template_name.endswith(".blend"):
        template_name = template_name + ".blend"

    p = join(directory_internal(), template_name)
    return p


def file_blend_slab_simulation_template(template_name: str = None) -> str:
    """Path to slab simulation template Blender file found in directory 'Internal/'.

    The .blend extension is added automatically if not given.

    :param template_name: Name of the template file. If None given,
        the default template is used.
    """

    if template_name is None:
        template_name = C.filename_slab_sim_forest_template

    if not template_name.endswith(".blend"):
        template_name = template_name + ".blend"

    p = join(directory_internal(), template_name)
    return p


def file_blend_system_simulation(simulation_name: str) -> str:
    """Path to a certain system simulation scene Blender file.

    The directory and the actual file share the same name but the file has a .blend
    extension.

    :param simulation_name: Name of the template file.
    """

    if not simulation_name.endswith(".blend"):
        blend_file_name = simulation_name + ".blend"
        simulation_directory_name = simulation_name
    else:
        blend_file_name = simulation_name
        simulation_directory_name = simulation_name[:-6]

    p = join(directory_system_simulation(simulation_directory_name), blend_file_name)
    return p


def file_spectral_cube(system_sim_name: str, file_type: str) -> str:
    """Path to simulated spectral image cube header file.

    The header file is ENVI format convention that contains the metadata of the cube.

    File type parameter as accepted by
    :func:`data.file_names.filename_system_sim_spectral_cube()`
    """

    p = join(
        directory_system_spectral_cube(system_sim_name),
        FN.filename_system_sim_spectral_cube(system_sim_name, file_type=file_type),
    )
    return p


def file_system_sim_preview(system_sim_name: str, image_name: str) -> str:
    """Path to system simulation preview render files.

    :param system_sim_name: Name of the system simulation.
    :param image_name: Name of the image. Use the image names available in
        :mod:`src.constants`. Image type postfix is added automatically.
    """

    if not image_name.endswith(C.postfix_plot_image_format):
        image_name = image_name + C.postfix_plot_image_format

    p = join(directory_system_sim_rend(system_sim_name), image_name)
    return p


def file_system_slab_csv(system_sim_name: str, slab_material_name: str) -> str:
    """Spectral slab material csv file name.

    It contains the spectral parameters of the slab material in a csv format
    that rendering scripts can read directly.
    """

    p = join(
        directory_system_simulation(system_sim_name),
        FN.filename_slab_material_csv(slab_material_name),
    )
    return p


def file_system_sim_light_spectra_csv(
    system_sim_name: str, light_file_name: str
) -> str:
    """Path to the light spectra csv file that is used for rendering.

    Specific for system_simulation type system simulation.
    """

    if not light_file_name.endswith(".csv"):
        light_file_name = light_file_name + ".csv"

    p = join(directory_system_simulation(system_sim_name), light_file_name)
    return p


def file_forest_soil_csv(system_sim_name: str) -> str:
    """Path to the soil spectra csv file that is used for rendering.

    Specific for forest type system simulation.

    TODO: change the logic so that this is a common call to any kind of reflective
        material and not specific to soil.
    """

    p = join(directory_system_simulation(system_sim_name), "blender_soil.csv")
    return p


def file_gsv_soil_dry_vector() -> str:
    """Soil dry vector used by GSV."""

    # this can be hard-coded as it is only for GSV usage. Might add to constants though.
    p = join(directory_code_soil(), "DryVec.txt")
    return p


def file_gsv_soil_humid_vector() -> str:
    """Soil humid vector used by GSV."""

    # this can be hard-coded as it is only for GSV usage. Might add to constants though.
    p = join(directory_code_soil(), "SMVec.txt")
    return p


def file_system_sim_rgb_colors_csv(system_sim_name: str) -> str:
    """Path to the RGB colors csv file that is used in rendering as a color approximation."""

    p = join(directory_system_simulation(system_sim_name), "rgb_colors.csv")
    return p


def file_visibility_map(system_sim_name: str, file_name: str) -> str:
    """Path to a visibility map file.

    :param system_sim_name: Name of the system simulation.
    :param file_name: Name of the visibility map file. The file name is used as-is.
    """

    p = join(
        directory_system_rend_visibility_maps(system_sim_name=system_sim_name),
        file_name,
    )
    return p


def find_visibility_map(system_sim_name: str, search_term: str) -> str:
    """Find a visibility map matching given search term.

    Search term should be something like "Leaf material 1" or "Trunk material 2".
    For white reference paths, one can use convenience method find_reference_visibility_map()
    that only needs reflectance as an identifier.

    :param system_sim_name: System simulation name.
    :param search_term: Search term that is included in a file name. Does not have to be
        a full match to the filename. We cannot fully control the filenames coming out of
        Blender, so we only check if the filename includes the search term instead of a full
        match.

    :return: Returns a path to the file.

    :raises KeyError: if more than one file match the search term.
    :raises FileNotFoundError: if no file match the search term.
    """

    file_names = []
    p = directory_system_rend_visibility_maps(system_sim_name=system_sim_name)

    for filename in os.listdir(p):
        if search_term in filename:
            file_names.append(filename)

    n = len(file_names)

    if n > 1:
        raise KeyError(
            f"Found more than one ({n}) file containing the search term '{search_term}' "
            f"in directory {p}. Available visibility maps: {list_visibility_maps(system_sim_name)}"
        )
    if n < 1:
        raise FileNotFoundError(
            f"Could not find a visibility map file containing the search term '{search_term}' "
            f"in directory {p}. Available visibility maps: {list_visibility_maps(system_sim_name)}"
        )

    res = join(p, file_names[0])
    return res


def find_reference_visibility_map(system_sim_name: str, reflectivity: float) -> str:
    """Convenience method for finding reference visibility map file.

    Calls :func:`find_visibility_map()`.

    :param system_sim_name: System simulation name.
    :param reflectivity: Reflectivity desired between 0.0 and 1.0. Must be one of the
        available ones in visibility maps directory of the scene.

    :return: Returns a path to the file.
    """

    # TODO: to filenaming
    search_term = f"Reference {reflectivity:.2f} material"
    return find_visibility_map(system_sim_name=system_sim_name, search_term=search_term)


def list_visibility_maps(system_sim_name: str) -> list[str]:
    """Lists all available visibility maps for given system_simulation scene.

    :param system_sim_name: Name of the system simulation.

    :return: List of visibility map file names.
    """

    p = directory_system_rend_visibility_maps(system_sim_name=system_sim_name)
    return os.listdir(p)


def list_reference_visibility_maps(system_sim_name: str) -> list[str]:
    """Lists all available reference visibility maps for given system simulation scene."""

    p = directory_system_rend_visibility_maps(system_sim_name=system_sim_name)
    res = [filename for filename in os.listdir(p) if "Reference" in filename]
    return res


def join(*args) -> str:
    """Custom join function to avoid problems using os.path.join.

    :param args: List of strings (directory names and a possibly a file name as
        the last element) to be joined.
    """

    n = len(args)
    s = ""
    for i, arg in enumerate(args):
        if i == n - 1:
            s = s + arg
        else:
            s = s + arg + "/"
    p = os.path.abspath(s)
    return p


def path_nn_model(slab_model_name: str = None) -> str:
    """Returns path to the NN model.

    Existence of the file has to be checked by caller.

    :param slab_model_name: Name of the slab model directory. If none given,
        the NN model in the default slab model directory is returned.
    """

    if slab_model_name is None:
        model_dir = directory_slab_model()
    else:
        model_dir = directory_slab_model(slab_model_name=slab_model_name)

    model_path = join(model_dir, C.slab_nn_name)

    return model_path


def directory_slab_rend_reference(imaging_type: str, base_path: str) -> str:
    """Returns the path to reflectance or transmittance reference folder of the slab simulation.

    :param imaging_type: String either 'refl' for reflectance or 'tran' for transmittance.
        Use the ones listed in constants.py.
    :param base_path: Path to  the working folder. Usually the one returned by
        path_directory_slab_optimization_working_temp() is correct and other paths
        should only be used for testing and debugging.
    """

    if imaging_type == C.imaging_type_refl:
        p = join(base_path, C.folder_rend_ref_refl)
    elif imaging_type == C.imaging_type_tran:
        p = join(base_path, C.folder_rend_ref_tran)
    else:
        raise Exception(
            f"Imaging type {imaging_type} not recognized. Use "
            f"{C.imaging_type_refl} or {C.imaging_type_tran}."
        )
    return p


def find_slab_opt_render_by_wl(
    wl: float, mode: str, imaging_type: str, base_path: str
) -> str:
    """Search for slab simulation optimization render by wavelength.

    :param wl: Wavelength to be searched. Must match the image name with two decimals.
    :param mode: String either 'slab' or 'reference'.
        Use the ones listed in :mod:`src.constants`.
    :param imaging_type: String either 'refl' for reflectance or 'tran' for
        transmittance. Use the ones listed in :mod:`src.constants`.
    :param base_path: Path to the image folder. Usually the one returned by
        :func:`directory_slab_optimization_working()` is correct and other paths
        should only be used for testing and debugging.

    :returns: Returns absolute path to the image.

    :raises FileNotFoundError: if the image cannot be found
    """

    def almost_equals(f1: float, f2: float, epsilon=0.01):
        """Custom float equality for our desired 2 decimal accuracy."""

        res = abs(f1 - f2) <= epsilon
        return res

    # First, find the correct directory
    if mode == C.target_type_slab:
        directory = join(base_path, C.dirname_slab_sim_rend)
    elif mode == C.target_type_ref:
        directory = directory_slab_rend_reference(imaging_type, base_path)
    else:
        raise AttributeError(
            f"Target type must be either {C.target_type_slab} or "
            f"{C.target_type_ref}. Was {mode}."
        )

    # Go through the files in the directory and find the one with the correct wavelength
    for filename in os.listdir(directory):
        image_wl = FN.parse_wl_from_filename(filename)

        if almost_equals(wl, image_wl):
            image_name = FN.filename_slab_sim_render_refl_or_tran(imaging_type, wl)

            if mode == C.target_type_slab:
                return join(base_path, C.dirname_slab_sim_rend, image_name)

            elif mode == C.target_type_ref:
                return join(
                    directory_slab_rend_reference(imaging_type, base_path),
                    image_name,
                )

    raise FileNotFoundError(f"Could not find '{wl}' nm image from '{directory}'.")


def find_light_file(file_name: str, system_sim_name: str = None) -> str:
    """Attempts to find a light file with given filename.

    :param file_name: A file with this name is searched from `Light spectra` directory. If
        also ``system_sim_name`` is given, the system simulation directory is searched before
        extending the search to `Light spectra` directory.
    :param system_sim_name: Optional. If not given, system_simulation scene directory is
        not searched.
    :return: Path to found file.
    :raises FileNotFoundError: If the file is not found.
    """

    if system_sim_name is not None:
        logging.info(
            f"Trying to find lighting data from system_simulation scene directory "
            f"'{directory_system_simulation(system_sim_name)}'."
        )
        p = join(directory_system_simulation(system_sim_name), file_name)
        if os.path.exists(p):
            logging.info(f"Light data found.")
            return p
        else:
            logging.debug(
                f"Could not find light data from scene directory '{p}'. "
                f"Now searching default directory."
            )

    p_dir = directory_light_spectra()

    p = join(p_dir, file_name)
    if os.path.exists(p):
        return p

    raise FileNotFoundError(f"Could not find light file from '{p}'.")

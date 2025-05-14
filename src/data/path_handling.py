"""
Paths to directories and files.
"""

import os

from src import constants as C
from src.data import file_names as FN


##########################################################################
# Paths to directories
##########################################################################


# Top level directories


def path_directory_project_root():
    """Path to project root directory."""

    p = os.path.abspath(C.path_project_root)
    return p


def path_directory_internal() -> str:
    """Path to directory containing HyperBlend internal files."""

    p = join(C.path_project_root, C.dirname_internal)
    return p


def path_directory_system_simulation_top() -> str:
    """Top level system simulation directory."""

    p = join(C.path_project_root, C.dirname_system_sim)
    return p


def path_directory_light_spectra() -> str:
    """Path to light spectra directory that contains all spectra for light sources."""

    p = join(C.path_project_root, C.dirname_light_spectra)
    return p


def path_directory_reflectance_spectra() -> str:
    """Path to reflectance spectra directory that contains all spectra for
    materials that are only reflective - not transmittive and not light."""

    p = join(C.path_project_root, C.dirname_reflectance_spectra)
    return p


# Code directories


def path_source_code() -> str:
    """Path to source code directory."""

    p = join(C.path_project_root, C.dirname_source)
    return p


def path_directory_blender_scripts() -> str:
    """Blender scripts directory."""

    p = join(path_source_code(), C.dirname_blender_scripts)
    return p


def path_directory_definitions() -> str:
    """HyperBlend's internal definitions and constants."""

    p = join(path_source_code(), C.dirname_definitions)
    return p


def path_directory_soil_code() -> str:
    """Soil code directory that contain gsv spectra vectors used for gsv generation."""

    p = join(path_source_code(), C.dirname_gsv)
    return p


# Simulation directories


def path_directory_slab_model(solver_name: str = None) -> str:
    """Path to the arbitrary slab model directory where the
    surface model parameters and the neural network are stored.

    :param solver_name: If None, the default slab model is used, i.e.,
        :func:`path_directory_default_slab_model()` is called.
    """

    if solver_name is None:
        p = join(
            C.path_project_root, C.dirname_slab_models, C.dirname_slab_models_default
        )
    else:
        p = join(C.path_project_root, C.dirname_slab_models, solver_name)
    return p


def path_directory_slab_simulation_top() -> str:
    """Path to top level leaf measurement sets root folder."""

    p = join(C.path_project_root, C.dirname_slab_simulation)
    return p


def path_directory_slab_simulation(slab_sim_name: str) -> str:
    """Path to a specific slab simulation.

    'root/Slab simulation/<slab_sim_name>'
    """

    p = join(path_directory_slab_simulation_top(), slab_sim_name)
    return p


def path_directory_result_signal_top(set_name: str) -> str:
    """Path to top level directory where signals resulting from slab simulation are saved."""

    p = join(path_directory_slab_simulation(set_name), C.dirname_result_signal)
    return p


def path_directory_result_signal(set_name: str, sample_id: int) -> str:
    """Path to a directory where a specific signal resulted from slab simulation is saved."""
    # TODO the name should come from the name handler
    p = join(
        path_directory_result_signal_top(set_name),
        f"{C.signal_directory_prefix}_{sample_id}",
    )
    return p


def path_directory_target(set_name: str) -> str:
    """Path to the top level target signal directory."""

    p = join(path_directory_slab_simulation(set_name), C.dirname_opt_target_signal)
    return p


def path_directory_slab_optimization_working_temp(set_name: str, sample_id: int) -> str:
    """Path to top level working folder of the slab simulation."""

    p = join(
        path_directory_result_signal(set_name, sample_id),
        C.dirname_slab_sim_working_temp,
    )
    return p


def path_directory_optimization_result(set_name: str, sample_id: int) -> str:
    """Path to optimization result directory. Only used if slab model is run in optimization mode."""

    p = join(
        path_directory_result_signal(set_name, sample_id),
        C.dirname_optimization_results,
    )
    return p


def path_directory_slab_temp_rend(set_name: str, sample_id: int) -> str:
    """Path to slab simulation rendering directory."""

    p = join(
        path_directory_slab_optimization_working_temp(set_name, sample_id),
        C.dirname_slab_sim_rend,
    )
    return p


def path_directory_slab_working_refl_or_trans(
    rendering_target: str, imaging_type: str, base_path: str
) -> str:
    """Returns a path to correct directory according to given target and imaging type.

    :param base_path:
        Path to  the working folder. Usually the one returned by
        path_directory_slab_optimization_working_temp() is correct and other paths
        should only be used for testing and debugging.
    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param rendering_target:
        String either 'slab' or 'reference'. Use the ones listed in constants.py.
    :raises AttributeError: If wrong target type is given.
    """

    if rendering_target == C.target_type_slab:
        return join(base_path, C.dirname_slab_sim_rend)
    elif rendering_target == C.target_type_ref:
        return path_directory_slab_rend_reference(imaging_type, base_path)
    else:
        raise AttributeError(
            f"Target type must be either {C.target_type_slab} or "
            f"{C.target_type_ref}. Was {rendering_target}."
        )


def path_directory_slab_rend_reference(imaging_type: str, base_path: str) -> str:
    """Returns the path to reflectance or transmittance reference folder of the slab simulation.

    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param base_path:
        Path to  the working folder. Usually the one returned by
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


def path_directory_system_simulation(forest_id: str) -> str:
    """Specific system simulation scene directory."""

    # TODO the name should come from the name handler
    p = join(path_directory_system_simulation_top(), f"scene_{forest_id}")
    return p


def path_directory_forest_rend(forest_id: str) -> str:
    """Rendering directory of the system simulation."""

    p = join(path_directory_system_simulation(forest_id), C.dirname_system_sim_rend)
    return p


def path_directory_system_spectral_cube(forest_id: str) -> str:
    """Spectral cube resulting from a system simulation is stored here."""

    p = join(path_directory_system_simulation(forest_id), C.dirname_system_cube)
    return p


def path_directory_system_rend_spectral(forest_id: str) -> str:
    """System simulation spectral rend directory."""

    p = join(path_directory_forest_rend(forest_id), C.dirname_system_sim_spectral_rend)
    return p


def path_directory_system_rend_visibility_maps(forest_id: str) -> str:
    """Rend directory for visibility maps of materials."""

    p = join(
        path_directory_forest_rend(forest_id), C.dirname_system_sim_visibility_maps_rend
    )
    return p


##########################################################################
# Paths to files
##########################################################################


def path_file_surface_model_parameters(solver_dirname: str = None) -> str:
    """Path to surface model parameter file.

    :param solver_dirname:
        Name of the solver directory. If none given, the default surface model directory is used.
    """

    if solver_dirname is None:
        p = join(path_directory_slab_model(), C.slab_surf_name)
    else:
        p = join(
            path_directory_slab_model(solver_name=solver_dirname), C.slab_surf_name
        )

    return p


def path_file_wl_result(set_name: str, wl: float, sample_id: int) -> str:
    """Path to wavelength result toml file of given sample and wavelength."""

    p = join(
        path_directory_optimization_result(set_name, sample_id),
        FN.filename_wl_result(wl),
    )
    return p


def path_file_signal_result(slab_sim_name: str, signal_id: int) -> str:
    """Path to signal result toml file of given slab simulation."""

    p = join(
        path_directory_result_signal(slab_sim_name, signal_id),
        FN.filename_sample_result(signal_id),
    )
    return p


def path_file_signal_result_plot(slab_sim_name: str, signal_id: int) -> str:
    """Path to signal result plot file of given slab simulation."""

    p = join(
        path_directory_result_signal(slab_sim_name, signal_id),
        FN.filename_signal_result_plot(signal_id),
    )
    return p


def path_file_slab_sim_result(slab_sim_name: str) -> str:
    """Path to slab simulation result toml file of given slab simulation."""

    filename = C.filename_slab_sim_result + C.postfix_text_data_format
    p = join(path_directory_slab_simulation(slab_sim_name=slab_sim_name), filename)
    return p


def path_file_slab_sim_result_plot(slab_sim_name: str) -> str:
    """Path to slab simulation result plot file of given slab simulation."""

    filename = C.filename_slab_sim_result + C.postfix_plot_image_format
    p = join(path_directory_slab_simulation(slab_sim_name=slab_sim_name), filename)
    return p


def path_file_slab_sim_error_plot(slab_sim_name: str) -> str:
    """Path to slab simulation error plot file of given slab simulation."""

    filename = C.filename_slab_sim_error_plot + C.postfix_plot_image_format
    p = join(path_directory_slab_simulation(slab_sim_name=slab_sim_name), filename)
    return p


def path_file_target(set_name: str, sample_id: int, resampled=False):
    """Path to leaf measurement target spectrum file of given slab simulation and signal.

    :param resampled:
        If True, path to corresponding resampled file is returned instead. Default is False.
    """

    p = join(
        path_directory_target(set_name),
        FN.filename_target(sample_id, resampled=resampled),
    )
    return p


def path_file_spectral_sampling(set_name: str):
    """Path to spectral resampling data of given set."""

    p = join(
        path_directory_target(set_name),
        C.file_sampling_data + C.postfix_text_data_format,
    )
    return p


def path_file_starting_guess(solver_name=None):
    """Path to the starting guess parameter file to be used in optimization.

    :param solver_name: If None given the default starting guess is used.
    """

    filename = "starting_guess" + C.postfix_text_data_format
    p = join(path_directory_slab_model(solver_name=solver_name), filename)
    return p


def path_file_rendered_image(
    target_type: str, imaging_type: str, wl: float, base_path: str
) -> str:
    """Returns a full path to an image of given wavelength.

    :param target_type:
        String either 'slab' or 'reference'. Use the ones listed in constants.py.
    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance.
        Use the ones listed in constants.py.
    :param wl:
        Wavelength.
    :param base_path:
        Path to  the working folder. Usually the one returned by
        get_path_opt_working() is correct and other paths
        should only be used for testing and debugging.
    :returns:
        Returns absolute path to the image.
    """

    image_name = FN.filename_rendered_image(imaging_type, wl)
    if target_type == C.target_type_slab:
        return join(base_path, C.dirname_slab_sim_rend, image_name)
    elif target_type == C.target_type_ref:
        return join(
            path_directory_slab_rend_reference(imaging_type, base_path), image_name
        )
    else:
        raise Exception(
            f"Target type must be either {C.target_type_slab} or {C.target_type_slab}. Was {target_type}."
        )


def path_system_simulation_template(
    template_name=C.filename_system_sim_forest_template,
):
    """Path to system simulation template blend file found in directory 'Internal/'.

    The .blend extension is added automatically if not given.

    :param template_name: Name of the template file. Default is 'system_sim_forest_template'.
    """

    if not template_name.endswith(".blend"):
        template_name = template_name + ".blend"

    p = join(path_directory_internal(), template_name)
    return p


def path_slab_simulation_template(template_name=C.filename_slab_sim_forest_template):
    """Path to slab simulation template blend file found in directory 'Internal/'.

    The .blend extension is added automatically if not given.

    :param template_name: Name of the template file. Default is 'slab_sim_template'.
    """

    if not template_name.endswith(".blend"):
        template_name = template_name + ".blend"

    p = join(path_directory_internal(), template_name)
    return p


def path_file_system_simulation_blend(simulation_name: str) -> str:
    """Path to a certain system simulation scene Blender file.

    The directory and the actual file share the same name but the file has a .blend
    extension.

    :param simulation_name: Name of the system simulation with or without .blend extension.
    """

    if not simulation_name.endswith(".blend"):
        blend_file_name = simulation_name + ".blend"
        simulation_directory_name = simulation_name
    else:
        blend_file_name = simulation_name
        simulation_directory_name = simulation_name[:-6]

    p = join(
        path_directory_system_simulation(simulation_directory_name), blend_file_name
    )
    return p


def path_file_system_sim_reflectance_header(forest_id: str) -> str:
    p = join(
        path_directory_system_spectral_cube(forest_id),
        FN.filename_system_sim_reflectance_header(forest_id),
    )
    return p


def path_file_system_sim_preview(system_sim_name: str, image_name: str):
    """Path to system simulation preview render files.

    Use the image names available in :mod:`src.constants`.
    """

    if not image_name.endswith(C.postfix_plot_image_format):
        image_name = image_name + C.postfix_plot_image_format

    p = join(path_directory_forest_rend(system_sim_name), image_name)
    return p


def path_file_system_slab_csv(forest_id: str, leaf_index):
    """Spectral slab material parameters csv file name."""

    p = join(
        path_directory_system_simulation(forest_id),
        FN.filename_leaf_material_csv(leaf_index),
    )
    return p


def path_file_system_forest_sun_spectra_csv(forest_id: str):
    """Path to the light spectra csv file that is used for rendering.

    Specific for forest type system simulation.
    """
    # TODO what to do with this hard-coded stuff??
    p = join(path_directory_system_simulation(forest_id), "blender_sun.csv")
    return p


def path_file_forest_sky_csv(forest_id: str):
    """Path to the sky spectra csv file that is used for rendering.

    Specific for forest type system simulation.
    """
    # TODO what to do with this hard-coded stuff??
    p = join(path_directory_system_simulation(forest_id), "blender_sky.csv")
    return p


def path_file_forest_soil_csv(forest_id: str):
    """Path to the soil spectra csv file that is used for rendering.

    Specific for forest type system simulation.
    """
    p = join(path_directory_system_simulation(forest_id), "blender_soil.csv")
    return p


def path_file_soil_dry_vector():
    """Soil dry vector used by GSV."""

    p = join(path_directory_soil_code(), "DryVec.txt")
    return p


def path_file_soil_humid_vector():
    """Soil humid vector used by GSV."""

    p = join(path_directory_soil_code(), "SMVec.txt")
    return p


def path_file_forest_rgb_csv(forest_id: str):
    p = join(path_directory_system_simulation(forest_id), "rgb_colors.csv")
    return p


def path_file_visibility_map(forest_id: str, file_name: str):
    p = join(path_directory_system_rend_visibility_maps(forest_id=forest_id), file_name)
    return p


def find_visibility_map(forest_id: str, search_term: str):
    """Find a visibility map matching given search term.

    Search term should be something like "Leaf material 1" or "Trunk material 2".
    For white reference paths, one can use convenience method find_reference_visibility_map()
    that only needs reflectance as an identifier.

    :param forest_id:
        Forest scene identifier.
    :param search_term:
        Search term that is included in a file name. Does not have to be a full match
        to the filename. We cannot fully control the filenames coming out of Blender,
        so we only check if the filename includes the search term instead of a full
        match.

    :return:
        Returns a path to the file.

    :raises KeyError: if more than one file match the search term.
    :raises FileNotFoundError: if no file match the search term.
    """

    file_names = []
    p = path_directory_system_rend_visibility_maps(forest_id=forest_id)

    for filename in os.listdir(p):
        if search_term in filename:
            file_names.append(filename)

    n = len(file_names)

    if n > 1:
        raise KeyError(
            f"Found more than one ({n}) file containing the search term '{search_term}' "
            f"in directory {p}. Available visibility maps: {list_visibility_maps(forest_id)}"
        )
    if n < 1:
        raise FileNotFoundError(
            f"Could not find a visibility map file containing the search term '{search_term}' "
            f"in directory {p}. Available visibility maps: {list_visibility_maps(forest_id)}"
        )

    res = join(p, file_names[0])
    return res


def find_reference_visibility_map(forest_id: str, reflectivity: float):
    """Convenience method for finding reference visibility map file.

    Calls find_visibility_map().

    :param forest_id:
        Forest scene identifier.
    :param reflectivity:
        Reflectivity desired between 0.0 and 1.0. Must be one of the available ones in
        visibility maps directory of the scene.
    :return:
        Returns a path to the file.
    """

    search_term = f"Reference {reflectivity:.2f} material"
    return find_visibility_map(forest_id=forest_id, search_term=search_term)


def list_visibility_maps(forest_id: str):
    """Lists all available visibility maps for given forest scene."""

    p = path_directory_system_rend_visibility_maps(forest_id=forest_id)
    return os.listdir(p)


def list_reference_visibility_maps(forest_id: str):
    """Lists all available reference visibility maps for given forest scene."""

    p = path_directory_system_rend_visibility_maps(forest_id=forest_id)
    res = [filename for filename in os.listdir(p) if "Reference" in filename]
    return res


def join(*args) -> str:
    """Custom join function to avoid problems using os.path.join."""

    n = len(args)
    s = ""
    for i, arg in enumerate(args):
        if i == n - 1:
            s = s + arg
        else:
            s = s + arg + "/"
    p = os.path.abspath(s)
    return p


def path_nn_model(solver_dirname: str = None):
    """Returns path to the NN model.

    Existence of the file has to be checked by caller.

    :param solver_dirname: Name of the solver directory. If none given, the default NN model is used.
    :return: Returns path to the NN model.
    """

    if solver_dirname is None:
        model_dir = path_directory_slab_model()
    else:
        model_dir = path_directory_slab_model(solver_name=solver_dirname)

    model_path = join(model_dir, C.slab_nn_name)

    return model_path

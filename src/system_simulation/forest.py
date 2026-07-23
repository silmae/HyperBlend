"""

This module contains system simulation functionality in case of a forest simulation.

.. note::
    If you plan to build a custom system simulation say, a conveyor belt with plastic
    bits, you should create a similar file to this as much of the stuff here is specific
    for a forest simulation. That is why this file is not simpy called something like
    `system_simulation.interface`. It is simply not so general.
"""

import logging

import numpy as np
import copy

from src.utils import spectra_utils as SU
from src.data import file_handling as FH, path_handling as PH, toml_handling as TH
import src.constants as C
from src.system_simulation import lighting
from src.system_simulation import soil
from src import plotter
from src.blender_scripts import forest_control
from src.blender_scripts import forest_constants as FC
from src.rendering import blender_control as BC
from src.setup.runtime_environment import RuntimeEnvironment
from src.data import cube_handling as CH


def init(
    leaves=None,
    soil_name: str = None,
    sun_file_name: str = None,
    sky_file_name: str = None,
    system_sim_name_to_copy_from: str = None,
    new_system_sim_name: str = None,
    conf_type: str = None,
    rng=None,
) -> str:
    """Create a new forest system simulation by copying a template.

    If you want to produce a non-working (i.e., you cannot actually run the system simulation)
    copy, do not provide any of the arguments. For a working copy, you have to produce
    at least the leaves. Other arguments can still be left to None and the default values
    will be used.

    Loads slab (leaf) material parameters for each leaf. They must have the same spectral
    sampling, but they do not have to originate from a single :term:`Slab simulation`.

    Loads sun and sky scatter spectra and resample them to match the range and resolution of
    the leaves. The spectra are normalized so that the brightest band of the brightest spectra
    will be unity, and the other spectrum is normalized with that too. In other words, the sky scatter
    may be brighter than the direct sunlight (overcast sky) and the sky spectrum would reach the value
    of 1 but the sun spectrum would not (unless they are equal). Same works if the sunlight is brighter
    than sky scattered light. These resampled spectra are saved locally for this system simulation for
    later use.

    :param leaves: Leaves should be given as list of tuples
        [(slab_simulation_name: str, sample_id: int, slab_material_name: str), (),...].
        If None, the scene will be copied, but no sun, sky, or leaf spectra will be copied.
        In this case, an empty string is returned.
    :param soil_name: Uses the soil spectrum from a file that includes this string. First found occurrence is used.
    :param sun_file_name: Name of the file with the sun spectrum to be used.
    :param sky_file_name: Name of the file with the sky spectrum to be used.
    :param system_sim_name_to_copy_from:
        If given, a system_simulation scene with this name will be copied instead of the default system_simulation template.
    :param new_system_sim_name:
        If given, this will be the name for the new system_simulation instead of the default generated name,
        which is meant for creating an arbitrary number of randomized clones.
    :param conf_type:
        How to produce configuration file: string on of "m2m", "m2s", or "s2m". The configuration file is most
        useful in randomization.

            - "m2m" (from master to master) makes a pure copy of the scene configuration file from the source scene.
              This is the default behavior.
            - "m2s" (from master to slave) will generate (gaussian) random values based on standard deviations defined in
              the source master configuration file.
            - "s2m" (from slave to master) will create a new master configuration from the source scene configuration
              with default standard deviation.

        .. note:: Configuration type s2s does not exist as there is no standard deviations present in slave configs.
    :param rng: Numpy random number generator for randomizing forest and tree parameters (the geometry of the scene).
        This is only needed if ``conf_type="m2m"`` and failing to provide it **will raise an error**.

    :return: New system_sim_name that is generated if new_system_sim_name is not given.
    """

    if system_sim_name_to_copy_from is not None:
        forest_id = FH.duplicate_system_simulation_scene(
            src_system_sim_name=system_sim_name_to_copy_from,
            dst_system_sim_name=new_system_sim_name,
        )
    else:
        forest_id = FH.duplicate_system_simulation_scene(
            dst_system_sim_name=new_system_sim_name
        )

    if system_sim_name_to_copy_from is not None:
        source_path = PH.directory_system_simulation(
            system_sim_name=system_sim_name_to_copy_from
        )
    else:
        source_path = PH.directory_internal()

    # Config file
    if conf_type is None or conf_type == "m2m":
        control_dict = forest_control.read_toml_as_dict(
            directory=source_path, filename=C.filename_system_sim_control
        )
        forest_control.write_forest_control(
            system_sim_name=forest_id, control_dict=control_dict
        )
    elif conf_type == "m2s":
        control_dict = forest_control.read_toml_as_dict(
            directory=source_path, filename=C.filename_system_sim_control
        )
        control_dict = _m2s(control_dict=control_dict, rng=rng)
        forest_control.write_forest_control(
            system_sim_name=forest_id, control_dict=control_dict
        )
    elif conf_type == "s2m":
        control_dict = forest_control.read_toml_as_dict(
            directory=source_path, filename=C.filename_system_sim_control
        )
        control_dict = _s2m(control_dict=control_dict)
        forest_control.write_forest_control(
            system_sim_name=forest_id, control_dict=control_dict
        )
    else:
        raise AttributeError(
            f"Attribute conf_type '{conf_type}' not recognised. Use one of ['m2m','m2s','s2m']."
        )

    # forest_id = '0102231033' # for debugging and testing

    if leaves is None:
        logging.info(
            f"No leaves were provided for system_simulation initialization, so I just copied the system_simulation scene."
        )
        return ""

    # load requested leaf sample result dicts
    sample_list = []
    for leaf in leaves:
        set_name = leaf[0]
        sample_id = leaf[1]
        sample_res = TH.read_signal_result(slab_sim_name=set_name, signal_id=sample_id)
        sample_list.append(sample_res)

    # Check that all leaves have been solved with the same sampling
    sampling = sample_list[0][C.key_sample_result_wls]
    logging.info(f"Checking that leaves' spectral band counts and wavelengths match.")
    for i, sample in enumerate(sample_list):

        wls_other = sample_list[i][C.key_sample_result_wls]
        other_set_name = leaves[i][0]
        other_sample_id = leaves[i][1]
        reference_set_name = leaves[0][0]
        reference_sample_id = leaves[0][1]

        # Check band count
        if len(sampling) != len(wls_other):
            raise ValueError(
                f"Band count for set '{other_set_name}' sample {other_sample_id} "
                f"(len = {len(wls_other)}) does not match "
                f"{reference_set_name} sample {reference_sample_id} (len = {len(sampling)}).\n"
            )
        # Check wavelengths
        same = np.allclose(sampling, wls_other, atol=0.01)
        if not same:
            raise ValueError(
                f"Wavelengths for {other_set_name} sample {other_sample_id} does not match "
                f"{reference_set_name} sample {reference_sample_id}.\n "
                f"Expected {sampling}\n"
                f"but got {wls_other}"
            )

    # Write leaf params
    logging.info(f"Bands and wavelengths ok. Copying leaf data.")
    for leaf in leaves:
        set_name = leaf[0]
        sample_id = leaf[1]
        leaf_id = leaf[2]
        FH.copy_slab_material_parameters(
            system_sim_name=forest_id,
            slab_material_name=leaf_id,
            src_slab_sim_name=set_name,
            signal_id=sample_id,
        )

    ################ Leaf RGB ################

    rgb_dict = {}

    # Define false colors
    for i, sample in enumerate(sample_list):
        wls = sample[C.key_sample_result_wls]
        refl = sample[C.key_sample_result_r]
        rgb = SU.spectra_to_rgb(wls=wls, value=refl)

        leaf_id = leaves[i][2]
        dict_key = f"LRGB_{leaf_id}"
        rgb_dict[dict_key] = rgb

    # print(f"RGB dict '{rgb_dict}'.")
    FH.write_blender_rgb_colors(system_sim_name=forest_id, rgb_dict=rgb_dict)

    ################ Loading Sun and Sky ################

    # TODO Loading and normalizing sun and sky needs to be checked and normalization reworked.

    # TODO Why do we load twice? There might be an actual reason for this, so refactor very carefully.
    logging.info(f"Loading sun data from file '{sun_file_name}'.")
    sun_wls_org, sun_irradiance_org = lighting.load_light(
        file_name=sun_file_name, system_sim_name=forest_id, lighting_type="sun"
    )
    logging.info(f"Reloading sun with new sampling with file '{sun_file_name}'.")
    sun_wls, sun_irradiance = lighting.load_light(
        file_name=sun_file_name,
        system_sim_name=forest_id,
        sampling=sampling,
        lighting_type="sun",
    )

    # TODO Load sky here

    # TODO Then normalize them together as sky can be brighter than the sun. This should be tested too.
    #       No time to implement this but it should be done.

    logging.info(f"Normalizing sun spectrum.")
    # Normalizing sun
    sun_irr_max = np.max(sun_irradiance)
    sun_irradiance = sun_irradiance / sun_irr_max
    FH.write_blender_light_spectra(
        system_sim_name=forest_id,
        wls=sun_wls,
        irradiances=sun_irradiance,
        lighting_type="sun",
    )

    logging.info(f"Plotting sun spectrum.")
    plotter.plot_light_data(
        wls=sun_wls_org,
        irradiances=sun_irradiance_org,
        wls_binned=sun_wls,
        irradiances_binned=sun_irradiance,
        forest_id=forest_id,
        lighting_type="sun",
        light_plot_name=sun_file_name,
    )

    ################ Sky ################

    logging.info(f"Loading sky spectrum from file '{sky_file_name}'.")
    sky_wls_org, sky_irradiance_org = lighting.load_light(
        file_name=sky_file_name, system_sim_name=forest_id, lighting_type="sky"
    )
    logging.info(f"Reloading sky with new sampling with file '{sky_file_name}'.")
    sky_wls, sky_irradiance = lighting.load_light(
        file_name=sky_file_name,
        system_sim_name=forest_id,
        sampling=sampling,
        lighting_type="sky",
    )
    logging.info(f"Normalizing sky spectrum.")
    # Normalize with maximum SUN irradiance
    sky_irradiance = sky_irradiance / sun_irr_max
    FH.write_blender_light_spectra(
        system_sim_name=forest_id,
        wls=sky_wls,
        irradiances=sky_irradiance,
        lighting_type="sky",
    )
    logging.info(f"Plotting sky spectrum.")
    plotter.plot_light_data(
        wls=sky_wls_org,
        irradiances=sky_irradiance_org,
        wls_binned=sky_wls,
        irradiances_binned=sky_irradiance,
        forest_id=forest_id,
        light_plot_name=sky_file_name,
        lighting_type="sky",
    )

    ################ Soil ################

    if soil_name is None:
        soil_name = "median_humid_clay"
        logging.warning(
            f"Soil name not provided for system_simulation initialization. Using default soil '{soil_name}'."
        )

    soil_wls, soil_refls = soil.load_soil(forest_id=forest_id, soil_name=soil_name)
    soil_wls_resampled, soil_refls_resampled = soil.load_soil(
        forest_id=forest_id, soil_name=soil_name, sampling=sampling
    )
    FH.write_blender_soil(
        system_sim_name=forest_id,
        wls=soil_wls_resampled,
        reflectances=soil_refls_resampled,
    )
    plotter.plot_blender_soil(
        wls=soil_wls,
        reflectances=soil_refls,
        soil_name=soil_name,
        wls_resampled=soil_wls_resampled,
        reflectances_resampled=soil_refls_resampled,
        forest_id=forest_id,
        dont_show=True,
        save=True,
    )

    return forest_id


def create_scene_bundle(
    bundle_name: str,
    system_sim_name_ancestor: str,
    rng,
    count=10,
    leaves=None,
    soil_name: str = None,
    sun_file_name: str = None,
    sky_file_name: str = None
):
    """Creates a bundle of scene variants and initializes them.

    Loop count times and create slave scenes with initialization.
    Gather and write generated scene names into a toml.
    Return path to the toml so it can be run in a bundle
    by :py:func:`system_simulation.forest.run_scene_bundle`.

    Parameters without explanation are passed directrly to
    :py:func:`system_simulation.forest.init`.

    :param bundle_name:
        Name of the bundle (file name).
    :param system_sim_name_ancestor:
        Which scene is to be used as an ancestor for the new scenes.
        It must be a master scene (indicated in the scene control file).
    :param rng:
        Numpy random number generator.
    :param count:
        How many scene variations are to be created.

    :return:
        The path to the created bundle file.
    """

    list_scene_names = []
    for i in range(count):
        new_scene_name = init(
            system_sim_name_to_copy_from=system_sim_name_ancestor,
            rng=rng,
            leaves=leaves,
            conf_type="m2s",
            soil_name=soil_name,
            sun_file_name=sun_file_name,
            sky_file_name=sky_file_name,
        )
        list_scene_names.append(new_scene_name)

    bundle_dict = {"list_scene_names": list_scene_names}
    p = TH.write_sys_sim_bundle(bundle_name=bundle_name, sys_sim_bundle_dict=bundle_dict)
    return p


def run_scene_bundle(runtime: RuntimeEnvironment, bundle_name: str, slab_material_names, render_spectral=True, render_visibility_maps=True, render_preview=True, construct_cube=True):
    """Runs the scene bundle either by rendering or by constructing the spectral image cube.

    All of these can be done with a single call.

    See other parameters from :py:func:`system_simulation.forest.render_forest` .

    :param runtime:
        See :term:`runtime`.
    :param bundle_name:
        Name of the bundle (file name).
    :param slab_material_names:
        Slab material name list as accepted by
        :py:func:`system_simulation.forest.setup_forest_for_rendering`.
    :param render_spectral:
        If True, renders spectral bands.
    :param render_visibility_maps:
        If True, renders visibility maps (needed for spectral cube construction).
    :param render_preview:
        If True, renders preview images.
    :param construct_cube:
        If True, constructs the spectral cube out of rendered frames.
        Cannot be called before spectral bands and visibility maps are rendered.
    """

    bundle_dict = TH.read_sys_sim_bundle(bundle_name=bundle_name)
    list_scene_names = bundle_dict["list_scene_names"]
    for _, system_sim_name in enumerate(list_scene_names):

        if slab_material_names is None:
            raise RuntimeError(f"Slab material names must be provided.")

        setup_forest_for_rendering(runtime=runtime, system_sim_name=system_sim_name, slab_material_names=slab_material_names)

        if render_preview:
            render_forest(runtime=runtime, system_sim_name=system_sim_name, render_mode="preview")
        if render_visibility_maps:
            render_forest(runtime=runtime, system_sim_name=system_sim_name, render_mode="visibility")
        if render_spectral:
            render_forest(runtime=runtime, system_sim_name=system_sim_name, render_mode="spectral")
        if construct_cube:
            construct_spectral_cube(system_sim_name=system_sim_name)


def process_forest_control(
    runtime: RuntimeEnvironment, system_sim_name: str, generate=True
):
    """Either (re)generates the control file (if `generate=True`) or applies it to the scene (if `generate=False`).

    This should be called after making any changes to the Blender scene file manually,
    i.e., using Blender directly rather than calling the HyperBlend's internal scripts
    to reflect those changes in the control file. And in reverse, if you manually change
    the control file, this method will apply the changes to the scene file (.blend).

    :param runtime: See :term:`runtime`.
    :param system_sim_name: See :term:`system_sim_name`.
    :param generate: If True, generate the control file based on the scene file. If False,
        does the opposite, i.e., applies the control file to the scene file.
    """

    BC.process_forest_control(
        runtime=runtime,
        system_sim_name=system_sim_name,
        global_master=False,
        generate=generate,
    )


def setup_forest_for_rendering(
    runtime: RuntimeEnvironment, system_sim_name: str, slab_material_names=None
):
    """Set up the system simulation for rendering.

    Most importantly, this applies the material and light values frame by frame to
    reflect the values needed to each spectral band.

    .. warning:: If this is not called before rendering, the results are arbitrary.

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param system_sim_name: Name of the system simulation to be set up.
    :param slab_material_names: Names of the slab materials (must mach the ones used in the
        Blender scene) as a list of strings like: ['Slab material 1', 'Slab material 2',...].
    """

    BC.setup_system_sim_scene(
        runtime, system_sim_name, slab_material_names=slab_material_names
    )


def render_forest(
    runtime: RuntimeEnvironment, system_sim_name: str, render_mode: str, silent=True
):
    """Renders the forest scene into a spectral image.

    :param runtime: See :term:`runtime`.
    :param system_sim_name: See :term:`system_sim_name`.
    :param render_mode:
        One of the following 'preview', 'spectral' or 'visibility'.
        'preview' renders only some preview images that can give an idea of the
        scene geometry without having to open the Blender scene itself.
        'spectral' renders all spectral bands (one rendered image per band).
        'visibility' renders visibility maps that show which object is visible
        in each pixel. This **must** be done before you can construct the reflectance
        image cube as the white correction relies on this information to find the
        reference plate locations.
    :param silent:
        If True, Blender output is redirected to null stream to avoid
        cluttering of console.
    """

    BC.render_forest(runtime, system_sim_name, render_mode, silent=silent)


def construct_spectral_cube(
    system_sim_name: str, system_sim_name_for_white_signal: str = None
):
    """Constructs an ENVI-style hyperspectral image cube out of rendered images.

    Can be used after the scene has been rendered (at least spectral and visibility maps).

    White reference for reflectance calculation is searched automatically from
    available visibility maps if `system_sim_name_for_white_signal` is not given .
    Note that the maps must be named like `Reference 0.00 material_0001.tif`.

    Saves white signal used in reflectance calculation as a toml file.

    Default RGB bands for ENVI metadata are inferred if in visible range.
    Otherwise first, middle, and last bands are used.

    :param system_sim_name: See :term:`system_sim_name`.
    :param system_sim_name_for_white_signal: Optionally, give system simulation name
        from where to fetch the white signal. This is useful when you have a lot of scenes
        with the same illumination, so the white signal does not have to be inferred for
        every scene separately. If None, it is inferred from the data.
        If there are no visibility maps in the current system simulation, this will fail
        and raise an error.

    :raises FileNotFoundError: if the rendered frames directory does not exist or is
        empty. Also, if the sun data file does not exist, which is needed for wavelength info.
    """

    CH.construct_envi_cube(
        system_sim_name=system_sim_name,
        system_sim_name_for_white_signal=system_sim_name_for_white_signal,
    )


def _m2s(control_dict: dict, rng, is_seed=False) -> dict:
    """Rewrites given master control dictionary into a slave control dictionary.

    New values are drawn from Gaussian distribution based on standard deviations present in
    the control. New random seeds (for the Blender file) are drawn from uniform distribution.

    :param control_dict:
        Master control dictionary.
    :param rng:
        Numpy random generator object used to randomize values in the slave control file.
    :param is_seed:
        If True, new value is drawn from discrete uniform distribution.

    :return:
        New slave control dictionary.

    :raises
        RuntimeError if given dict is not master control.
    """

    new_dict = {}

    for key, dict_item in control_dict.items():

        if key == FC.key_ctrl_is_master_control and dict_item is False:
            raise RuntimeError(
                f"Cannot apply randomness from a slave system_simulation control file."
            )
        elif key == FC.key_ctrl_is_master_control and dict_item is True:
            # Change the control file type from master to slave.
            new_value = False
        elif isinstance(dict_item, dict):
            # Recursion for sub-dictionaries.
            # Seed is a special variable as we take it from normal distribution so let's mark it for next recursion
            is_seed = "Seed" in key
            new_value = _m2s(control_dict=dict_item, rng=rng, is_seed=is_seed)
        elif key == FC.key_ctrl_item_std:
            # If one of the keys is 'Standard deviation', we are inside a sub-dictionary that is
            #   essentially a single value that we must randomize.
            return _gaussian(control_dict, rng=rng, is_seed=is_seed)
        else:
            new_value = dict_item

        new_dict[key] = new_value

    return new_dict


def _gaussian(param_dict: dict, rng, is_seed=False) -> dict:
    """Applies gaussian random value to a value that is represented as a dictionary in system_simulation control file.

    :param param_dict:
        Dict representation of the value.
    :param rng:
        A Numpy random generator object to be used for randomization.
    :param is_seed:
        If True, new value is drawn from discrete uniform distribution.
    :return:
        A new dict with randomized value based on standard deviation that was
        present in given dict. The std field is removed from returned dict representation.
    """

    new_dict = copy.deepcopy(param_dict)

    std = abs(param_dict[FC.key_ctrl_item_std])
    value = param_dict[FC.key_ctrl_item_value]
    item_type = param_dict[FC.key_ctrl_item_type]

    if is_seed:
        new_val = int(rng.integers(0,10000))
    elif item_type == "INT":
        new_val = int(rng.normal(loc=value, scale=std))
    elif item_type == "VALUE":  # float
        new_val = rng.normal(loc=value, scale=std)

    del new_dict[FC.key_ctrl_item_std]

    new_dict[FC.key_ctrl_item_value] = new_val
    return new_dict


def _s2m(control_dict: dict) -> dict:
    """Change slave control into master control.

    Basically, just adds default STD to each item that needs it.
    Almost opposite operation of m2s().

    :param control_dict:
        Slave dict to be masterfied. If given control is already master,
        it will be returned without modifications.

    :return:
        New master control dict.
    """

    new_dict = copy.deepcopy(control_dict)

    for key, dict_item in control_dict.items():

        if key == FC.key_ctrl_is_master_control and dict_item is True:
            logging.info(
                f"Already a master control file. Doing nothing and returning the original control."
            )
            return control_dict
        elif key == FC.key_ctrl_is_master_control and dict_item is False:
            # Change the control file type from slave to master.
            new_value = True
        elif isinstance(dict_item, dict):
            if key == "Seed":
                new_value = dict_item
            else:
                # Recursion for sub-dictionaries.
                new_value = _s2m(control_dict=dict_item)
        elif key == FC.key_ctrl_item_type:
            item_type = control_dict[FC.key_ctrl_item_type]
            if item_type == "INT" or item_type == "VALUE":
                internal_value = control_dict[FC.key_ctrl_item_value]
                new_dict[FC.key_ctrl_item_std] = (
                    internal_value * FC.ctrl_default_std_of_value
                )
        else:
            new_value = dict_item

        new_dict[key] = new_value

    return new_dict

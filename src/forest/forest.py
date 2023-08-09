
import logging

import os
import numpy as np
import shutil

from src.leaf_model.opt import Optimization
from src.utils import spectra_utils as SU
from src.data import file_handling as FH, path_handling as PH, toml_handling as TH
import src.constants as C
from src.forest import lighting
from src.forest import soil
from src import plotter
from src.blender_scripts import forest_control


def init(leaves=None, soil_name: str = None, sun_file_name: str = None, sky_file_name: str = None,
         copy_forest_id: str = None, custom_forest_id: str = None, conf_type: str = None):
    """

    Create a new forest by copying template.

    Load leaf material parameters for each leaf. They must use same spectral sampling,
    but do not have to be from a single measurement set.

    Load sun and resample its spectra to match the leaves.
    Normalize so that highest intensity is 1.
    Save as local sun spectra.

    Load sky and resample its spectra to match the leaves.
    Normalize with highest sun intensity.
    Save as local sky spectra.

    # TODO Load trunk reflectance spectrum.

    :param soil_name:
    :param leaves:
        Leaves should be given as list of tuples [(set_name: str, sample_id: int, leaf_material_name: str), (),...].
    :param sun_file_name:
    :param sky_file_name:
    :param copy_forest_id:
        If given, a forest scene with this id will be copied instead of the default forest template.
    :param custom_forest_id:
        If given, this will be the identifier for the new forest instead of the standard generated id.
    :param conf_type:
        How to produce configuration file: string from ['m2m','m2s','s2m'].
            - m2m (from master to master) makes a pure copy of the scene configuration file from the source scene.
              This is the default behavior.
            - m2s (from master to slave) will generate (gaussian) random values based on standard deviations defined in
              the source master configuration file.
            - s2m (from slave to master) will create a new master configuration from the source scene configuration
              with default standard deviation.
        Note: s2s does not exist as there is no standard deviations present in slave configs.
    :return
        Forest id that is generated if custom_forest_id is not given.
    """

    if copy_forest_id is not None:
        forest_id = FH.duplicate_forest_scene(copy_forest_id=copy_forest_id, custom_forest_id=custom_forest_id)
    else:
        forest_id = FH.duplicate_forest_scene(custom_forest_id=custom_forest_id)

    if copy_forest_id is not None:
        source_path = PH.path_directory_forest_scene(forest_id=copy_forest_id)
    else:
        source_path = PH.path_directory_project_root()

    # Config file
    if conf_type is None or conf_type == 'm2m':
        control_dict = forest_control.read_toml_as_dict(directory=source_path, filename=C.file_forest_control)
        forest_control.write_forest_control(forest_id=forest_id, control_dict=control_dict)
    elif conf_type == 'm2s':
        control_dict = forest_control.read_toml_as_dict(directory=source_path, filename=C.file_forest_control)
        # TODO apply randomness
        forest_control.write_forest_control(forest_id=forest_id, control_dict=control_dict)
    elif conf_type == 's2m':
        control_dict = forest_control.read_toml_as_dict(directory=source_path, filename=C.file_forest_control)
        # TODO add default standard deviation
        forest_control.write_forest_control(forest_id=forest_id, control_dict=control_dict)
    else:
        raise AttributeError(f"Attribute conf_type '{conf_type}' not recognised. Use one of ['m2m','m2s','s2m'].")


    # forest_id = '0102231033' # for debugging and testing

    if leaves is None:
        logging.info(f"No leaves were provided for forest initialization, so I just copied the forest scene.")
        return

    # load requested leaf sample result dicts
    sample_list = []
    for leaf in leaves:
        set_name = leaf[0]
        sample_id = leaf[1]
        sample_res = TH.read_sample_result(set_name=set_name,sample_id=sample_id)
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
            raise ValueError(f"Band count for set '{other_set_name}' sample {other_sample_id} (len = {len(wls_other)}) does not match "
                             f"{reference_set_name} sample {reference_sample_id} (len = {len(sampling)}).\n")
        # Check wavelengths
        same = np.allclose(sampling, wls_other, atol=0.01)
        if not same:
            raise ValueError(f"Wavelengths for {other_set_name} sample {other_sample_id} does not match "
                             f"{reference_set_name} sample {reference_sample_id}.\n "
                             f"Expected {sampling}\n"
                             f"but got {wls_other}")

    # Write leaf params
    logging.info(f"Bands and wavelengths ok. Copying leaf data.")
    for leaf in leaves:
        set_name = leaf[0]
        sample_id = leaf[1]
        leaf_id = leaf[2]
        FH.copy_leaf_material_parameters(forest_id=forest_id, leaf_id=leaf_id, source_set_name=set_name, sample_id=sample_id)

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
    FH.write_blender_rgb_colors(forest_id=forest_id, rgb_dict=rgb_dict)

    ################ Sun ################

    logging.info(f"Normalizing, resampling and writing sun data.")
    sun_wls_org, sun_irradiance_org = lighting.load_light(file_name=sun_file_name, forest_id=forest_id, lighting_type='sun')
    logging.info(f"Reloading sun with new sampling.")
    sun_wls, sun_irradiance = lighting.load_light(file_name=sun_file_name, forest_id=forest_id, sampling=sampling, lighting_type='sun')
    # Normalizing sun
    sun_irr_max = np.max(sun_irradiance)
    sun_irradiance = sun_irradiance / sun_irr_max
    FH.write_blender_light_spectra(forest_id=forest_id, wls=sun_wls, irradiances=sun_irradiance, lighting_type='sun')

    logging.info(f"Plotting sun data.")
    plotter.plot_light_data(wls=sun_wls_org, irradiances=sun_irradiance_org, wls_binned=sun_wls, irradiances_binned=sun_irradiance,
                            forest_id=forest_id, lighting_type='sun')

    ################ Sky ################

    sky_wls_org, sky_irradiance_org = lighting.load_light(file_name=sky_file_name, forest_id=forest_id, lighting_type='sky')
    sky_wls, sky_irradiance = lighting.load_light(file_name=sky_file_name, forest_id=forest_id, sampling=sampling, lighting_type='sky')
    # Normalize with maximum SUN irradiance
    sky_irradiance = sky_irradiance / sun_irr_max
    FH.write_blender_light_spectra(forest_id=forest_id, wls=sky_wls, irradiances=sky_irradiance, lighting_type='sky')
    plotter.plot_light_data(wls=sky_wls_org, irradiances=sky_irradiance_org, wls_binned=sky_wls, irradiances_binned=sky_irradiance, forest_id=forest_id,
                            sun_plot_name=sky_file_name, lighting_type='sky')

    ################ Soil ################

    if soil_name is None:
        soil_name = "median_humid_clay"
        logging.warning(f"Soil name not provided for forest initialization. Using default soil '{soil_name}'.")

    soil_wls, soil_refls = soil.load_soil(forest_id=forest_id, soil_name=soil_name)
    soil_wls_resampled, soil_refls_resampled = soil.load_soil(forest_id=forest_id, soil_name=soil_name, sampling=sampling)
    FH.write_blender_soil(forest_id=forest_id, wls=soil_wls_resampled, reflectances=soil_refls_resampled)
    plotter.plot_blender_soil(wls=soil_wls, reflectances=soil_refls, soil_name=soil_name, wls_resampled=soil_wls_resampled,
                              reflectances_resampled=soil_refls_resampled, forest_id=forest_id, dont_show=True, save=True)

    return forest_id


import logging

import os
import numpy as np

from src.leaf_model.opt import Optimization
from src.utils import spectra_utils as SU
from src.data import file_handling as FH, path_handling as PH, toml_handling as TH
import src.constants as C
from src.forest import sun
from src import plotter


def init(leaves, sun_file_name: str = None):
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

    # TODO Load ground reflectance spectrum.

    # TODO Load trunk reflectance spectrum.
    """

    # forest_id = FH.duplicate_scene_from_template()
    forest_id = '0102231033' # for debugging and testing

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

    logging.info(f"Normalizing, resampling and writing sun data.")
    sun_wls_org, sun_irradiance_org = sun.load_sun(file_name=sun_file_name, forest_id=forest_id)
    logging.info(f"Reloading sun with new sampling.")
    sun_wls, sun_irradiance = sun.load_sun(file_name=sun_file_name, forest_id=forest_id, sampling=sampling)
    # Normalizing sun
    irr_max = np.max(sun_irradiance)
    sun_irradiance = sun_irradiance / irr_max
    FH.write_blender_sun_spectra(forest_id=forest_id, wls=sun_wls, irradiances=sun_irradiance)

    logging.info(f"Plotting sun data.")
    plotter.plot_sun_data(wls=sun_wls_org, irradiances=sun_irradiance_org, wls_binned=sun_wls, irradiances_binned=sun_irradiance, forest_id=forest_id)

    # bands, wls, irradiances = FH.read_blender_sun_spectra(forest_id=forest_id)



def leaf_csv_name(name, resolution):
    name = f'{name}_res_{resolution:.0f}'
    return name

# TODO delete
# def do_leaves(resolution=50):
#
#     logging.info(f"Generating leaf data if it does not exist yet.")
#
#     set_name = leaf_csv_name('normal', resolution)
#     if not os.path.exists(PH.path_directory_set_result(set_name)):
#         wls,r,t = prospect.get_default_prospect_leaf()
#         SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
#         o = Optimization(set_name=set_name)
#         o.run_optimization(resolution=resolution, prediction_method='surface')
#
#     set_name = leaf_csv_name('dry', resolution)
#     if not os.path.exists(PH.path_directory_set_result(set_name)):
#         wls, r, t = prospect.get_default_prospect_leaf_dry()
#         SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
#         o = Optimization(set_name=set_name)
#         o.run_optimization(resolution=resolution, prediction_method='surface')


def generate_some_leaf_stuff(scene_id ="0123456789", resolution=50):


    if not os.path.exists(PH.path_file_forest_scene(scene_id)):
        raise RuntimeError(f"Blend file {PH.path_file_forest_scene(scene_id)} does not exist. Cannot generate leafs.")

    do_leaves(resolution)
    logging.info(f"Copying leaf data to forest '{scene_id}'")
    FH.copy_leaf_material_parameters(scene_id, 1, leaf_csv_name('normal', resolution))
    FH.copy_leaf_material_parameters(scene_id, 2, leaf_csv_name('dry', resolution))
    FH.copy_leaf_material_parameters(scene_id, 3, leaf_csv_name('dry', resolution))

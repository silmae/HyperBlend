
import logging

import os
from src.prospect import prospect

from src.leaf_model.opt import Optimization
from src.utils import spectra_utils as SU
from src.data import file_handling as FH
from src.data import path_handling as PH


def leaf_csv_name(name, resolution):
    name = f'{name}_res_{resolution:.0f}'
    return name

def do_leaves(resolution=50):

    logging.info(f"Generating leaf data if it does not exist yet.")

    set_name = leaf_csv_name('normal', resolution)
    if not os.path.exists(PH.path_directory_set_result(set_name)):
        wls,r,t = prospect.get_default_prospect_leaf()
        SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
        o = Optimization(set_name=set_name)
        o.run_optimization(resolution=resolution, prediction_method='surface')

    set_name = leaf_csv_name('dry', resolution)
    if not os.path.exists(PH.path_directory_set_result(set_name)):
        wls, r, t = prospect.get_default_prospect_leaf_dry()
        SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
        o = Optimization(set_name=set_name)
        o.run_optimization(resolution=resolution, prediction_method='surface')


def generate_some_leaf_stuff(scene_id ="0123456789", resolution=50):


    if not os.path.exists(PH.path_file_forest_scene(scene_id)):
        raise RuntimeError(f"Blend file {PH.path_file_forest_scene(scene_id)} does not exist. Cannot generate leafs.")

    do_leaves(resolution)
    logging.info(f"Copying leaf data to forest '{scene_id}'")
    FH.copy_leaf_material_parameters_from(leaf_csv_name('normal', resolution), scene_id, 1)
    FH.copy_leaf_material_parameters_from(leaf_csv_name('dry', resolution), scene_id, 2)
    FH.copy_leaf_material_parameters_from(leaf_csv_name('dry', resolution), scene_id, 3)

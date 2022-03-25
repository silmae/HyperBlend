
import logging

import sys
import numpy as np
import os
from src.prospect import prospect

from src.optimization import Optimization
from src.surface_model import surface_model as SM
from src.utils import spectra_utils as SU
from src.data import file_handling as FH
from src.data import path_handling as PH
from src.rendering import blender_control as BC


def do_leaves():

    logging.info(f"Generating leaf data if it does not exist yet.")

    set_name = 'p_normal'
    if not os.path.exists(PH.path_directory_set_result(set_name)):
        wls,r,t = prospect.get_default_prospect_leaf()
        SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
        o = Optimization(set_name=set_name)
        o.run_optimization(resolution=100, prediction_method='surface')

    set_name = 'p_dry'
    if not os.path.exists(PH.path_directory_set_result(set_name)):
        wls, r, t = prospect.get_default_prospect_leaf_dry()
        SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
        o = Optimization(set_name=set_name)
        o.run_optimization(resolution=100, prediction_method='surface')


def generate_some_leaf_stuff(forest_id = "0123456789"):


    if not os.path.exists(PH.path_file_forest_scene(forest_id)):
        raise RuntimeError(f"Blend file {PH.path_file_forest_scene(forest_id)} does not exist. Cannot generate leafs.")

    do_leaves()
    logging.info(f"Copying leaf data to forest '{forest_id}'")
    FH.copy_leaf_material_parameters_from('p_normal', forest_id, 1)
    FH.copy_leaf_material_parameters_from('p_dry', forest_id, 2)
    FH.copy_leaf_material_parameters_from('p_dry', forest_id, 3)

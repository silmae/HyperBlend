"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import logging
import os.path
import datetime

import os
import numpy as np
import matplotlib.pyplot as plt

import plotter
import src.leaf_model.nn
from src.leaf_model import surf as SM
from src.data import path_handling as PH
from src.forest import forest
from src.blender_scripts import forest_control as control

from src.data import toml_handling as TH
from src.data import cube_handling as CH
from src.leaf_model import interface as LI
from src import constants as C
from src.reflectance_lab import diffuse_reflectance

from src.rendering import blender_control as BC
from src.gsv import gsv
from src.forest import soil


def write_forest_control(forest_id: str, control_dict: dict):
    TH.write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_forest_scene(forest_id=forest_id), filename='forest_control')


def read_forest_control(forest_id: str) -> dict:
    return TH.read_toml_as_dict(directory=PH.path_directory_forest_scene(forest_id=forest_id), filename='forest_control')


def forest_pipe_test(rng):

    # Generating low resolution random leaves
    set_name = 'low_res_w_dry'
    new_sampling = [450,500,550,600,700,800]
    # LI.generate_prospect_leaf_random(set_name=set_name, leaf_count=2)
    # LI.generate_prospect_leaf(set_name=set_name, sample_id=4) # add one dry default leaf
    # LI.resample_leaf_targets(set_name=set_name, new_sampling=new_sampling)
    # LI.solve_leaf_material_parameters(set_name=set_name, clear_old_results=True)
    #
    # leaves = [(set_name, 0, 'Leaf material 1'), (set_name, 1, 'Leaf material 2'), (set_name, 3, 'Leaf material 3')]
    # forest_id = forest.init(leaves=leaves, conf_type='s2m', rng=rng, copy_forest_id='1208231030')

    """
    Running forest.init only copies files. Running setup makes the Blender scene renderable.
    """

    forest_id = '1208231030'
    control_dict = control.read_forest_control(forest_id=forest_id)

    BC.setup_forest(scene_id=forest_id, base_sun_power=20, leaf_id_list=['Leaf material 1', 'Leaf material 2', 'Leaf material 3'])#, 'Leaf material 4'])
    # scene_dict = read_forest_control(forest_id=forest_id)
    # print("moi")

    # BC.render_forest(scene_id=forest_id, render_mode='preview')
    # BC.render_forest(scene_id=forest_id, render_mode='abundances')
    # BC.render_forest(scene_id=forest_id, render_mode='spectral')
    #
    # CH.construct_envi_cube(forest_id=forest_id)
    # CH.show_cube(forest_id=forest_id)

    # BC.generate_forest_control(global_master=True)


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    # logging.basicConfig(stream=sys.stdout, level='INFO')
    path_dir_logs = "../log"
    if not os.path.exists(path_dir_logs):
        os.makedirs(path_dir_logs)

    log_identifier = str(datetime.datetime.now())
    log_identifier = log_identifier.replace(' ', '_')
    log_identifier = log_identifier.replace(':', '')
    log_identifier = log_identifier.replace('.', '')

    log_file_name = f"{log_identifier}.log"
    log_path = PH.join(path_dir_logs, log_file_name)
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='w'),
                            logging.StreamHandler()
                        ])

    rng = np.random.default_rng(4321)

    # gsv.visualize_default_soils(save=False, dont_show=False)
    # gsv._write_default_soils()

    # plotter.plot_resampling(set_name='low_res')
    forest_pipe_test(rng=rng)
    # forest_id = forest.init()

    # forest_id = '1406231352'

    # Sun power test
    # diffuse_reflectance.run(data_exits=True)

    # leaf_stuff = [('try_random_p_leaves', 0, 1), ('try_random_p_leaves', 1, 3)]
    # forest.init(copy_forest_id='0102231033')

    # Let's first generate some random PROSPECT leaves
    # set_name = "try_random_p_leaves"
    # generates three leaf targets to \HyperBlend\leaf_measurement_sets\try_random_p_leaves\sample_targets
    # LI.generate_prospect_leaf_random(set_name=set_name, count=3)
    # Solve renderable leaf material parameters that produce target reflectance and transmittance
    # LI.solve_leaf_material_parameters(set_name=set_name, solver='nn', clear_old_results=True)
    # After solver has run, check results from HyperBlend\leaf_measurement_sets\try_random_p_leaves\set_result

    # sampling = [450, 550, 600.0, 650, 700, 750, 800.1, 900.0]
    # TH.write_sampling(set_name, sampling=sampling)
    # sampling = TH.read_sampling(set_name)
    # print(sampling)

    # LI.resample_leaf_targets(set_name=set_name)
    # plotter.plot_resampling(set_name=set_name)

    # # Similarly, we can provide exact parameters
    # set_name = "try_p_leaves"
    # # generates a leaf target with certain parameters to \HyperBlend\leaf_measurement_sets\try_p_leaves\sample_targets.
    # # The values used here are the default values.
    # LI.generate_prospect_leaf(set_name=set_name, sample_id=0, n=1.5, ab=32, ar=8, brown=0, w=0.016, m=0.009, ant=0)
    # # You can also give only some parameters. Defaults will be used for the ones not provided.
    # # Remember to give new sample_id so that the previously created leaf is not overwritten.
    # LI.generate_prospect_leaf(set_name=set_name, sample_id=1, w=0.001, m=0.03)
    # # Solve renderable leaf material parameters as before
    # LI.solve_leaf_material_parameters(set_name=set_name, resolution=10, solver='nn')
    # # After solver has run, check results from HyperBlend\leaf_measurement_sets\try_p_leaves\set_result
    #
    #
    # # We can also copy existing set and solve it with a different solver for example. Let's try that with
    # #   surface fitting solver called 'surf'
    # copy_set = "try_copying_set"
    # LI.solve_leaf_material_parameters(set_name=copy_set, resolution=10, solver='surf', copyof="try_p_leaves")
    #
    #
    # # Let's try manually creating some data to work with
    # set_name = "try_manual_set"
    # # Example data list of lists where inner list holds the data ordered as [wavelength, reflectance, transmittance]
    # data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    # # Write data to disk in a format the HyperBlend can understand
    # TH.write_target(set_name, data, sample_id=0, resampled=False)
    # # Solve as before
    # LI.solve_leaf_material_parameters(set_name=set_name, resolution=1, solver='opt', clear_old_results=True)

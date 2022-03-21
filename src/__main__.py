"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE.
"""

import logging

import sys
import numpy as np
from src.prospect import prospect

from src.optimization import Optimization
from src.surface_model import surface_model as SM
from src.utils import spectra_utils as SU
from src.data import file_handling as FH
from src.data import path_handling as PH
from src.rendering import blender_control as BC


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # scene_id = FH.duplicate_scene_from_template()
    scene_id = "0123456789" # id for debugging
    BC.setup_forest(scene_id)
    BC.render_forest_previews(scene_id)

    # SM.train(do_points=False, num_points=50)
    # SM.fit_surface(show_plot=True, save_params=False)

    # set_name = 'specchio_5nm'
    # set_name = 'surface_test_predict'
    # # FH.clear_folder(PH.path_directory_subresult(set_name, 0))
    # o = Optimization(set_name)
    # wls,r,t = prospect.get_default_prospect_leaf()
    # diff = r-t
    # max_diff = diff.max()
    # print(f'max difference of r and t = {max_diff}')
    # SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
    # o.run_optimization(resolution=20, use_threads=True, prediction_method='surface')

    # # Test the software with hard coded data.
    # presets.optimize_default_target(spectral_resolution=50)
    #
    # # Example using "real" data
    # data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    # set_name = 'test_set'
    # o = Optimization(set_name)
    # TH.write_target(set_name, data, sample_id=0)
    # o.run_optimization()

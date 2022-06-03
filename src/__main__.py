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
from src.surface_model import neural

if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # Train new starting guess ##########
    # Add to readme?
    # SU.generate_starting_guess()
    # SU.fit_starting_guess_coefficients()
    #############################

    # ########## Show surfaces
    # SM.fit_surface(show_plot=True, save_params=False)

    # ######### REDO points and training
    SM.train(do_points=True, num_points=80)
    # neural.fit_nn(show_plot=True, save_params=True, epochs=150)
    ###################

    # ad, sd, ai, mf = neural.predict_nn([0.2,0.3], [0.24,0.27])
    # print(ad)
    # print(sd)
    # print(ai)
    # print(mf)

    # wls, r, t = prospect.get_default_prospect_leaf()
    # wls2, r2, t2 = prospect.get_default_prospect_leaf_dict()
    # wls_diff = np.fabs(wls - wls2)
    # r_diff = np.fabs(r - r2)
    # t_diff = np.fabs(t - t2)
    # print(np.max(wls_diff))
    # print(np.max(r_diff))
    # print(np.max(t_diff))

    prospect.make_random_leaf_targets(count=10)
    prospect.run_prospect_randoms_simulation()


    # set_name = 'specchio_surface'
    # set_name = 'surface_test_predict_2'

    # FH.clear_folder(PH.path_directory_subresult(set_name, 0))
    # o = Optimization(set_name)
    # wls,r,t = prospect.get_default_prospect_leaf()
    # diff = r-t
    # max_diff = diff.max()
    # print(f'max difference of r and t = {max_diff}')
    # SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='surface')

    # # Test the software with hard coded data.
    # presets.optimize_default_target(spectral_resolution=50)
    #
    # # Example using "real" data
    # data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    # set_name = 'test_set'
    # o = Optimization(set_name)
    # TH.write_target(set_name, data, sample_id=0)
    # o.run_optimization()

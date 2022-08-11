"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE.
"""

import logging
import os.path
import datetime

import sys
import numpy as np
from src.prospect import prospect

from src.optimization import Optimization
from src.surface_model import surface_model as SM
from src.utils import spectra_utils as SU
from src.data import file_handling as FH
from src.data import path_handling as PH
from src.data import toml_handling as TH
from src.surface_model import neural
from src import plotter

from src.surface_model import surface_model_shared as shared

def maxdiff(set_name="prospect_randoms"):

    # Find maximum r t difference of sample.
    res = TH.read_sample_result(set_name, 0)
    r = np.array(res['refls_modeled'])
    t = np.array(res['trans_modeled'])
    diff = np.abs(r-t)
    print(f"max diff : {diff.max()}")


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
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='w'),
                            logging.StreamHandler()
                        ])

    # plotter.plot_set_result("surface_train", dont_show=False, save_thumbnail=False)

    # shared.visualize_training_data_pruning(set_name = "surface_train_good_airange05")

    # Train new starting guess ##########
    # Add to readme?
    # SU.generate_starting_guess()
    # SU.fit_starting_guess_coefficients()
    #############################

    # ########## Show surfaces
    # SM.fit_surface(show_plot=True, save_params=True, plot_data_as_surface=False,  show_nn=False)
    SM.fit_surface(show_plot=True, save_params=False, plot_data_as_surface=False,  show_nn=True)

    # ######### REDO points and NN training
    # SM.train(do_points=True, num_points=20, maxdiff_rt=0.25)
    # neural.fit_nn(show_plot=True, save_params=True, epochs=300, batch_size=8, learning_rate=0.0004, split=0.2, patience=30)
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

    # prospect.make_random_leaf_targets(count=100)
    # prospect.run_prospect_randoms_simulation()

    # set_name = 'specchio_surf'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='surface')
    #
    # set_name = 'specchio_nn'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')
    #
    # set_name = 'specchio_opt'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='optimization')

    # set_name = 'specchio_nn_8layer_300epoch'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_8layer_early_stop'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_8layer_8batch'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_8layer_64batch'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_8layer_16batch'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_10k_points'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_smallnet_2batch'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_nn_smallnet_2batch_newstuff'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='nn')

    # set_name = 'specchio_surface_10k_points'
    # o = Optimization(set_name)
    # o.run_optimization(resolution=5, use_threads=True, prediction_method='surface')



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

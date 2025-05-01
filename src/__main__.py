"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import logging
import datetime

import os
import numpy as np

from playground.integration_test_like import forest_pipe_test
from src.data import path_handling as PH

from src.setup import initialization

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

    initialization.initialize()

    from src.slab_model.opt import Optimization
    from src.slab_model import interface as SI

    slab_sim_opt_name = "integration_test_opt_slabs"
    SI.solve_leaf_material_parameters(set_name=slab_sim_opt_name, solver='surf', clear_old_results=False)

    # rng = np.random.default_rng(4321)
    # #### RUN ME FOR TESTING FOREST CANOPY STUFF OUT. SEE THE COMMENTS IN THE METHOD BEFORE RUNNING!!! ######
    # forest_pipe_test(rng=rng)


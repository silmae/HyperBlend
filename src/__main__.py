import logging
import os.path

import numpy as np
import sys
import argparse  # to parse options for us and print a nice help message

from src import constants as C
from src import data_utils as DU
from src import utils
from src import optimization
from src.optimization import Optimization
from src import file_handling as FH
from src import toml_handlling as T
from src import plotter
from src import presets
from src import spectra_utils as SU
from src import specchio_data_parser as SDP

if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    set_name = 'specchio_20nm'
    # # SU.make_linear_test_target(set_name)
    # opt = Optimization(set_name, clear_subresults=False)
    # opt.run_optimization(set_name, resolution=20, use_threads=True)

    # plotter.plot_averaged_sample_errors(set_name, dont_show=False, save_thumbnail=True)
    # plotter.plot_averaged_sample_result(set_name, dont_show=True, save_thumbnail=True)
    T.write_final_result(set_name)

    # SDP.combine_pairs()


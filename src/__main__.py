"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE (or command line, but prepare to fix bugs).
"""

import logging
import os.path

import numpy as np
import sys
import argparse  # to parse options for us and print a nice help message

# A bunch of improrts to self-made files you may need.
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

    # Test the software with hard coded data.
    presets.optimize_default_target(spectral_resolution=50)

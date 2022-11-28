"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE.
"""

import logging
import os.path
import datetime

import os
import numpy as np
import matplotlib.pyplot as plt

import src.leaf_model.nn
from src.leaf_model import surf as SM
from src.data import path_handling as PH

from src.data import toml_handling as TH
from src.leaf_model import interface
from src import constants as C




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

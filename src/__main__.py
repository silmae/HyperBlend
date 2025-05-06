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

    initialization.initialize()

    # rng = np.random.default_rng(4321)
    # #### RUN ME FOR TESTING FOREST CANOPY STUFF OUT. SEE THE COMMENTS IN THE METHOD BEFORE RUNNING!!! ######
    # forest_pipe_test(rng=rng)


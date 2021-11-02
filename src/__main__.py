"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE (or command line, but prepare to fix bugs).
"""

import logging

import sys

# A bunch of improrts to self-made files you may need.
from src import presets
from src.optimization import Optimization
from data import toml_handlling as TH

if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # Test the software with hard coded data.
    presets.optimize_default_target(spectral_resolution=50)

    # Example using "real" data
    data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    set_name = 'test_set'
    o = Optimization(set_name)
    TH.write_target(set_name, data, sample_id=0)
    o.run_optimization()

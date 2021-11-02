"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE (or command line, but prepare to fix bugs).
"""

import logging

import sys

# A bunch of improrts to self-made files you may need.
from src import presets

if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # Test the software with hard coded data.
    presets.optimize_default_target(spectral_resolution=50)

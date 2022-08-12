"""
Some presets to be run for testing the program.
"""

from src.utils import spectra_utils as SU
from src.leaf_model.material_param_optimization import Optimization


def optimize_default_target(spectral_resolution=50):
    """ Run optimization for hard coded test target.

    Use this if you want to test the software, but you don't have any data available.

    :param spectral_resolution:
        How dense sampling of wavelengths you want. Resolution of 1 will run the
        optimization for all wavelengths. It will take some time (few hours).
    :return:
        None
    """

    set_name = 'default_target_test'
    o = Optimization(set_name)
    SU.make_default_target(set_name)
    o.run_optimization(use_threads=True, resolution=spectral_resolution, prediction_method='optimization')

"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE.
"""

import logging

import sys

from src import presets
from src.optimization import Optimization
from data import toml_handling as TH

if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    import prospect_d as PD
    import plotter as P
    import matplotlib.pyplot as plt

    l, r, t = PD.run_prospect(
    n = 1.5,
    cab = 32,
    car = 8,
    cbrown = 0.,
    cw = 0.016,
    cm = 0.009,
    ant = 0.0,
    nr = None, kab = None, kcar = None, kbrown = None, kw = None,
    km = None, kant = None, alpha = 40.)
    print('moi')

    fig, ax = plt.subplots()
    P._plot_refl_tran_to_axis(ax, r, t, l, x_label='wavelength', invert_tran=True)
    plt.show()

    # # Test the software with hard coded data.
    # presets.optimize_default_target(spectral_resolution=50)
    #
    # # Example using "real" data
    # data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    # set_name = 'test_set'
    # o = Optimization(set_name)
    # TH.write_target(set_name, data, sample_id=0)
    # o.run_optimization()

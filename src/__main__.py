"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import numpy as np

from src.setup import initialization
from src.slab_simulation import interface as SMI
from src.data import toml_handling as TH
from src.system_simulation import forest as F
from src.playground import integration_test_like as ITL
from src.rendering import blender_control as BC
from system_simulation import dev_actions

rng = np.random.default_rng(666)

from src.gsv import interface as gsvi

if __name__ == "__main__":

    runtime = initialization.initialize()
    soil_spectra = gsvi.simulate_gsv_soil(c1=0.528, c2=-0.011, c3=0.014, cSM=-0.129)
    gsvi.write_soil_spectra(reflectance_spectra=soil_spectra, filename="My new soil spectrum")

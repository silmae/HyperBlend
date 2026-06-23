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

if __name__ == "__main__":

    runtime = initialization.initialize()

    set_name = "try_manual_set"

    # Example data list of lists where inner list holds the data ordered as [wavelength, reflectance, transmittance]
    target = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]

    # Write data to disk in a format the HyperBlend can understand
    TH.write_target(set_name, target, signal_id=0)

    # Solve as before
    SMI.solve_slab_material_parameters(runtime=runtime, slab_sim_name=set_name)

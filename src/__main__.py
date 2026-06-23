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

    slab_sim_name = "prospect_slab"
    SMI.generate_prospect_leaf(
        slab_sim_name=slab_sim_name,
        signal_id=0,
        n=1.5,
        ab=32.0,
        ar=8.0,
        brown=0.0,
        w=0.016,
        m=0.009,
        ant=0.0,
    )
    SMI.solve_slab_material_parameters(
        runtime=runtime,
        slab_sim_name=slab_sim_name,
        range_start=400,
        range_end=1000,
        resolution=100,
    )

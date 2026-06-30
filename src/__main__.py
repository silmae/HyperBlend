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
    slab_sim_name = "tutorial_slab_simulation"
    system_sim_name = "tutorial_system_simulation_2"
    slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

    # Pack leaf data for system_simulation scene initialization.
    leaves = [
        (slab_sim_name, 0, slab_material_names[0]),
        (slab_sim_name, 1, slab_material_names[1]),
        (slab_sim_name, 2, slab_material_names[2]),
    ]

    # F.init(
    #     leaves=leaves,
    #     conf_type="m2m",
    #     new_system_sim_name=system_sim_name,
    #     soil_name="median_humid_clay",
    #     sun_file_name="default_sun",
    #     sky_file_name="default_sky",
    # )
    F.render_forest(
        runtime=runtime, system_sim_name=system_sim_name, render_mode="preview"
    )

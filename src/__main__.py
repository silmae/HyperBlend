"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import numpy as np

from src.setup import initialization
from src.slab_simulation import interface as SMI
from src.data import toml_handling as TH
from src.playground import integration_test_like as ITL
from src.rendering import blender_control as BC
from system_simulation import dev_actions

rng = np.random.default_rng(666)

from src.system_simulation import forest as F

if __name__ == "__main__":

    runtime = initialization.initialize()

    # system_sim_name = "My simulation"
    bundle_name = "My second bundle"
    slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

    # Pack leaf data for system_simulation scene initialization.
    # leaves = [
    #     (slab_sim_name, 0, slab_material_names[0]),
    #     (slab_sim_name, 1, slab_material_names[1]),
    #     (slab_sim_name, 3, slab_material_names[2]),
    # ]

    # F.create_scene_bundle(bundle_name=bundle_name, system_sim_name_ancestor=bundle_ancestor_name, rng=rng, count=3, leaves=leaves)
    F.run_scene_bundle(runtime=runtime, bundle_name=bundle_name, slab_material_names=slab_material_names)

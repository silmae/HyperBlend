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

    rng = np.random.default_rng(666)

    slab_sim_name = "slabs_for_system_test"
    # system_sim_name = "tutorial_system_simulation_2"
    slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

    # Pack leaf data for system_simulation scene initialization.
    leaves = [
        (slab_sim_name, 0, slab_material_names[0]),
        (slab_sim_name, 1, slab_material_names[1]),
        (slab_sim_name, 3, slab_material_names[2]),
    ]

    # F.init(
    #     leaves=leaves,
    #     conf_type="m2m",
    #     new_system_sim_name=system_sim_name,
    #     soil_name="median_humid_clay",
    #     sun_file_name="default_sun",
    #     sky_file_name="default_sky",
    # )
    # F.render_forest(
    #     runtime=runtime, system_sim_name=system_sim_name, render_mode="preview"
    # )

    test_sys_sim_name = "bundle ancestor"
    F.init(
        leaves=leaves,
        conf_type="m2m",
        new_system_sim_name=test_sys_sim_name,
    )
    # F.process_forest_control(runtime=runtime, system_sim_name=test_sys_sim_name,generate=False)

    bundle_name = "My second bundle"
    F.create_scene_bundle(bundle_name=bundle_name, system_sim_name_ancestor=test_sys_sim_name, rng=rng, count=3, leaves=leaves)
    F.run_scene_bundle(runtime=runtime, bundle_name=bundle_name, slab_material_names=slab_material_names, render_spectral=False, construct_cube=False, render_visibility_maps=False)

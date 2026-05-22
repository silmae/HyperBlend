"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import numpy as np

from src.setup import initialization
from src.slab_simulation import interface as SMI
from src.system_simulation import forest as F
from src.playground import integration_test_like as ITL
from src.rendering import blender_control as BC

if __name__ == "__main__":

    runtime = initialization.initialize()
    slab_sim_name = "tutorial_slab_simulation"
    SMI.generate_prospect_leaf_random(slab_sim_name=slab_sim_name, leaf_count=3)
    SMI.solve_slab_material_parameters(
        runtime=runtime,
        slab_sim_name=slab_sim_name,
        range_start=400,
        range_end=1000,
        resolution=100,
    )

    # System simulation beginner tutorial
    system_sim_name = "tutorial_system_simulation"
    slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

    rng = np.random.default_rng(12345)

    # Pack leaf data for system_simulation scene initialization.
    leaves = [
        (slab_sim_name, 0, slab_material_names[0]),
        (slab_sim_name, 1, slab_material_names[1]),
        (slab_sim_name, 3, slab_material_names[2]),
    ]

    F.init(
        leaves=leaves,
        conf_type="m2m",
        rng=rng,
        new_system_sim_name=system_sim_name,
        soil_name=soil_name,
        sun_file_name=sun_name,
        sky_file_name=sky_name,
    )

    # BC.generate_forest_control(runtime=runtime, global_master=True)

    # rng = np.random.default_rng(123324)
    # ITL.forest_pipe_test(runtime=runtime, rng=rng)

    # This can be used to run the iterative training
    # SMI.iterative_train(
    #     runtime=runtime,
    #     iterations=8,
    #     training_points=100,
    #     dry_run=False,
    # )

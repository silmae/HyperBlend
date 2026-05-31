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
from system_simulation import dev_actions

if __name__ == "__main__":

    runtime = initialization.initialize()
    slab_sim_name = "tutorial_slab_simulation"
    # SMI.generate_prospect_leaf_random(slab_sim_name=slab_sim_name, leaf_count=3)
    # SMI.solve_slab_material_parameters(
    #     runtime=runtime,
    #     slab_sim_name=slab_sim_name,
    #     range_start=400,
    #     range_end=2500,
    #     resolution=50,
    #     clear_old_results=True,
    # )

    # System simulation beginner tutorial
    system_sim_name = "tutorial_system_simulation"
    slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

    rng = np.random.default_rng(12345)

    # Pack leaf data for system_simulation scene initialization.
    leaves = [
        (slab_sim_name, 0, slab_material_names[0]),
        (slab_sim_name, 1, slab_material_names[1]),
        (slab_sim_name, 2, slab_material_names[2]),
    ]

    # F.init(
    #     leaves=leaves,
    #     conf_type="m2m",
    #     rng=rng,
    #     new_system_sim_name=system_sim_name,
    #     soil_name="meadian_humid_peat",
    # )

    # FIRST we go this far and check the new files

    # F.process_forest_control(
    #     runtime=runtime, system_sim_name=system_sim_name, generate=False
    # )

    # F.process_forest_control(
    #     runtime=runtime, system_sim_name=system_sim_name, generate=True
    # )

    # F.setup_forest_for_rendering(
    #     runtime=runtime,
    #     system_sim_name=system_sim_name,
    #     leaf_name_list=slab_material_names,
    # )
    # F.render_forest(
    #     runtime=runtime, system_sim_name=system_sim_name, render_mode="preview"
    # )
    F.render_forest(
        runtime=runtime, system_sim_name=system_sim_name, render_mode="visibility"
    )
    F.render_forest(
        runtime=runtime, system_sim_name=system_sim_name, render_mode="spectral"
    )
    F.construct_spectral_cube(system_sim_name=system_sim_name)

    # REGENRATE FOREST CONTROL FILE #########
    from src.system_simulation import dev_actions

    # dev_actions.generate_global_master_forest_control_from_template(runtime=runtime)
    # dev_actions.apply_global_master_forest_control_to_template(runtime=runtime)
    ####################################

    # rng = np.random.default_rng(123324)
    # ITL.forest_pipe_test(runtime=runtime, rng=rng)

    # This can be used to run the iterative training
    # SMI.iterative_train(
    #     runtime=runtime,
    #     iterations=8,
    #     training_points=100,
    #     dry_run=False,
    # )

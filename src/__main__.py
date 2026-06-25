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

    SMI.train_models(
        runtime=runtime,
        slab_sim_name_for_training="My training data",
        generate_data=False,
        dry_run=False,
        train_surf=True,
        train_nn=True,
        layer_count=5,
        layer_width=1000,
        epochs=300,
        batch_size=32,
        learning_rate=0.01,
        patience=30,
        split=0.1,
        train_points_per_dim=20,
        show_plot=False,
        solver_name_to_save=None,
        solver_name_to_use=None,
    )

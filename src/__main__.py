"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import numpy as np

from src.setup import initialization
from src.slab_model import interface as SMI
from src.playground import integration_test_like as ITL
from src.rendering import blender_control as BC

if __name__ == "__main__":

    runtime = initialization.initialize()

    # BC.generate_forest_control(runtime=runtime, global_master=True)

    rng = np.random.default_rng(123324)
    ITL.forest_pipe_test(runtime=runtime, rng=rng)

    # This can be used to run the iterative training
    # SMI.iterative_train(
    #     runtime=runtime,
    #     iterations=8,
    #     training_points=100,
    #     dry_run=False,
    # )

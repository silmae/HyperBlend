"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import numpy as np

from src.setup import initialization
from src.playground import dataset_paper
from src.slab_model import interface as SMI
from src.playground import integration_test_like as ITL
from src.rendering import blender_control as BC

if __name__ == "__main__":

    runtime = initialization.initialize()

    # BC.generate_forest_control(runtime=runtime, global_master=True)

    # rng = np.random.default_rng(123324)
    dataset_paper.run(runtime=runtime)

    # dataset_paper.calculate_abundances()

    # dataset_paper.separate_spectral_renders(delete_originals=True, do_copy=False)

    # This can be used to run the iterative training
    # SMI.iterative_train(
    #     runtime=runtime,
    #     iterations=8,
    #     training_points=100,
    #     dry_run=False,
    # )

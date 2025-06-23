"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

from src.setup import initialization
from src.slab_model import interface as SMI


if __name__ == "__main__":

    runtime = initialization.initialize()

    SMI.iterative_train(
        runtime=runtime,
        iterations=8,
        training_points=100,
        dry_run=False,
    )

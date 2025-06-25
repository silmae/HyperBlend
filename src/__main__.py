"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

from src.setup import initialization
from src.playground import dataset_paper
from src.slab_model import interface as SMI


if __name__ == "__main__":

    runtime = initialization.initialize()

    dataset_paper.run(runtime=runtime)

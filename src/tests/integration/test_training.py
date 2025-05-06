"""

This module will test the training of slab models.

Test plan:
    - Test the normal starting guess generation to a cusotm dir
    - Test iterative training
        - First round with just starting guess curfe
        - Following rounds with surf model
        - The last round trains surf and nn
    - Trained model does not to be good so just a few training points
        can be used
    - We are mostly just testing that everything can be run without errors

"""

import os
import unittest # needed for skipping tests
from shutil import rmtree
from unittest import TestCase

from src.slab_model import interface as SMI
from src.data import path_handling as PH, toml_handling as TH
from src.slab_model import training_data as TD


class Test(TestCase):

    def test_starting_guess(self):

        # Test the normal starting guess generation to a custom dir
        slab_sim_name = "test_starting_guess"
        new_solver_name = "test_solver"
        # TD.generate_starting_guess(slab_sim_name=slab_sim_name, solver_name=new_solver_name, step=100)

        from src import plotter

        plotter._plot_starting_guess_coeffs_fitting(set_name=slab_sim_name, solver_name=new_solver_name)
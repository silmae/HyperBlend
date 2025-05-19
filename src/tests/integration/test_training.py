"""

This module will test the training of slab models.

Test plan:
    - Test the normal starting guess generation to a custom dir
    - Test iterative training
        - First round with just starting guess
        - Following rounds with surf model
        - The last round trains surf and nn
    - Trained model does not to be good so just a few training points
        can be used
    - We are mostly just testing that everything can be run without errors

"""

import os
import unittest  # needed for skipping tests
from shutil import rmtree
from unittest import TestCase
import logging

from src.slab_model import interface as SMI
from src.data import path_handling as PH, toml_handling as TH
from src.slab_model import training_data as TD
from src.setup import initialization
from src import constants as C


# @unittest.skip("Skipping training test for now")
class TestTraining(TestCase):

    def setUp(self):
        self.runtime = initialization.initialize()

    def tearDown(self):
        logging.shutdown()

    # @unittest.skip("Skipping test for now")
    def test_starting_guess(self):

        # Test the normal starting guess generation to a custom dir
        slab_sim_name = "test_starting_guess"
        new_solver_name = "test_solver"
        TD.generate_starting_guess(
            slab_sim_name=slab_sim_name,
            solver_name=new_solver_name,
            step=100,
            runtime=self.runtime,
        )

    def test_iterative_training(self):

        iterations = 3
        SMI.iterative_train(
            runtime=self.runtime,
            iterations=iterations,
            training_points=10,
            dry_run=False,
        )

        model_name_base = "train_iter_"
        for i in range(3):
            model_name = f"{model_name_base}{i+1}"

            p = PH.directory_slab_model(model_name)
            self.assertTrue(
                os.path.exists(p), msg=f"Model directory {p} does not exist."
            )

            p = PH.join(PH.directory_slab_model(model_name), C.slab_surf_name)
            self.assertTrue(os.path.exists(p), msg=f"Model file {p} does not exist.")

            if i == 2:
                # Check if the nn model exists only for the last iteration
                p = PH.join(PH.directory_slab_model(model_name), C.slab_nn_name)
                self.assertTrue(
                    os.path.exists(p), msg=f"Model file {p} does not exist."
                )

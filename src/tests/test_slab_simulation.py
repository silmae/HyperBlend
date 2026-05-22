"""
Test slab simulation with different solvers.

"""

import os
import unittest  # needed for skipping tests
from shutil import rmtree
from unittest import TestCase
import logging
import numpy as np

from src.setup import initialization
from src.slab_simulation import interface as SMI
from src.data import path_handling as PH, toml_handling as TH
from src.utils import data_utils as DU


# @unittest.skip("Skipping test_slab_simulation for now")
class TestSlabs(TestCase):

    def setUp(self):
        self.runtime = initialization.initialize()

    def tearDown(self):
        logging.shutdown()

    # @unittest.skip("Temporarily skipping test_slab_simulation.")
    def test_advanced_solvers(self):
        self.run_advanced_solvers()

    # @unittest.skip("Skipping test_slab_optimization.")
    def test_slab_optimization(self):
        self.run_slab_optimization()

    def test_sampling(self):
        """Tests that the slab simulation sampling works as expected."""

        logging.info("Testing slab simulation sampling.")

        # Check that the sampling is empty before running the test
        slab_sim_name = "integration_test_sampling"

        SMI.generate_prospect_leaf_random(slab_sim_name=slab_sim_name)

        # Missing parameters
        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name)

        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name, resolution=5),

        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name, range_start=500),

        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name, range_end=500),

        # Min wl too small
        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(
                slab_sim_name=slab_sim_name, wls=[300, 400, 500]
            )
        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(
                slab_sim_name=slab_sim_name,
                range_start=300,
                range_end=2000,
                resolution=5,
            )

        # Max wl too large
        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(
                slab_sim_name=slab_sim_name, wls=[2000, 3000, 4000]
            )
        with self.assertRaises(AttributeError):
            SMI.resample_slab_sim_target(
                slab_sim_name=slab_sim_name,
                range_start=400,
                range_end=3000,
                resolution=5,
            )

        resampling_wls = [400, 500, 600]
        SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name, wls=resampling_wls)

        target = TH.read_target(
            slab_sim_name=slab_sim_name, signal_id=0, resampled=True
        )
        target_wls, _, _ = DU.unpack_target(target=target)
        target_wls = np.array(target_wls)
        self.assertTrue(np.all(target_wls == resampling_wls))

        start = 400
        stop = 600
        resolution = 100
        SMI.resample_slab_sim_target(
            slab_sim_name=slab_sim_name,
            range_start=start,
            range_end=stop,
            resolution=resolution,
        )

        target = TH.read_target(
            slab_sim_name=slab_sim_name, signal_id=0, resampled=True
        )
        target_wls, _, _ = DU.unpack_target(target=target)
        target_wls = np.array(target_wls)
        self.assertTrue(np.all(target_wls == resampling_wls))

        # Finally, test uneven sampling
        resampling_wls = [400, 500, 751]
        SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name, wls=resampling_wls)

        target = TH.read_target(
            slab_sim_name=slab_sim_name, signal_id=0, resampled=True
        )
        target_wls, _, _ = DU.unpack_target(target=target)
        target_wls = np.array(target_wls)
        self.assertTrue(np.all(target_wls == resampling_wls))

    # @unittest.skip("Skipping test_custom_solver.")
    def test_custom_solver(self):
        """Tests that the slab simulation run with a custom solver works as expected."""

        logging.info("Testing custom solver.")

        fake_solver_name = "Integration test custom solver"
        path_default_slab_model = PH.directory_slab_model()
        path_custom_slab_model = PH.directory_slab_model(
            slab_model_name=fake_solver_name
        )

        # os.rename(path_custom_slab_model, path_default_slab_model)
        # exit()

        # Check that the default slab model directory exists and the custom one does not
        self.assertTrue(os.path.exists(path_default_slab_model))
        self.assertFalse(os.path.exists(path_custom_slab_model))

        # Rename the default slab model to fake a custom model for testing
        os.rename(path_default_slab_model, path_custom_slab_model)

        # Surround all the rest to try catch so that we can revert the name change
        try:
            # Check that the custom slab model directory exists and the default one does not
            self.assertTrue(os.path.exists(path_custom_slab_model))
            self.assertFalse(os.path.exists(path_default_slab_model))

            # These should now not work as the default slab model directory is renamed
            self.assertRaises(FileNotFoundError, self.run_advanced_solvers)
            self.assertRaises(Exception, self.run_slab_optimization)

            # Finally, run the advanced solvers and slab optimization with the custom solver
            self.run_advanced_solvers(solver_dirname=fake_solver_name)
            self.run_slab_optimization(solver_dirname=fake_solver_name)

        except:
            # Revert the name change of the default slab model directory
            #   even if something goes wrong in the test so that testing
            #   doesn't break the actual structure.
            os.rename(path_custom_slab_model, path_default_slab_model)
            raise

        # In case we did not run into an exception earlier, revert the name change here
        os.rename(path_custom_slab_model, path_default_slab_model)

    def check_existence_of_common_files(self, slab_sim_name):
        """Check that required result files exist after running the slab simulation.

        Files common to optimizer, surface fitting and NN are checked here. Files specific to
        each solver are checked in their respective tests.
        """

        logging.info(
            f"Checking existence of common files in {PH.directory_slab_simulation(slab_sim_name)} directory."
        )

        path_slab_sim_result = PH.file_slab_sim_result(slab_sim_name=slab_sim_name)
        fail_msg = (
            f"Could not find the slab simulation result file at {path_slab_sim_result}."
        )
        self.assertTrue(os.path.exists(path_slab_sim_result), msg=fail_msg)
        logging.info(f"Found slab simulation result file at {path_slab_sim_result}.")

        path_slab_sim_result_plot = PH.file_slab_sim_result_plot(
            slab_sim_name=slab_sim_name
        )
        fail_msg = f"Could not find the slab simulation result plot file at {path_slab_sim_result_plot}."
        self.assertTrue(os.path.exists(path_slab_sim_result_plot), msg=fail_msg)
        logging.info(
            f"Found slab simulation result plot file at {path_slab_sim_result_plot}."
        )

        path_slab_sim_error_plot = PH.file_slab_sim_error_plot(
            slab_sim_name=slab_sim_name
        )
        fail_msg = f"Could not find the slab simulation error plot file at {path_slab_sim_error_plot}."
        self.assertTrue(os.path.exists(path_slab_sim_error_plot), msg=fail_msg)
        logging.info(
            f"Found slab simulation error plot file at {path_slab_sim_error_plot}."
        )

    def check_existance_of_signal_result_files(self, slab_sim_name, signal_ids):
        """Check that the result files for the signals exist.

        :param slab_sim_name: Name of the slab simulation.
        :param signal_ids: List of signal IDs (int) to check.
        """

        for signal_id in signal_ids:
            p = PH.file_signal_result(slab_sim_name=slab_sim_name, signal_id=signal_id)
            with self.subTest(p=p):
                error_msg = (
                    f"Result file for signal {signal_id} does not exist at path {p}."
                )
                self.assertTrue(os.path.exists(p), msg=error_msg)
                logging.info(f"Found result file for signal at {p}.")

            p = PH.file_signal_result_plot(
                slab_sim_name=slab_sim_name, signal_id=signal_id
            )
            with self.subTest(p=p):
                error_msg = (
                    f"Result plot for signal {signal_id} does not exist at path {p}."
                )
                self.assertTrue(os.path.exists(p), msg=error_msg)
                logging.info(f"Found result plot for signal at {p}.")

    def check_nn_and_surf_result_files(self, solver: str, solver_dirname: str = None):
        """Check that the result files for the neural network and surface fitting solvers exist.

        :param solver: Solver to check. Either 'nn' or 'surf'.
        """

        if solver == "nn":
            slab_sim_name = "integration_test_nn_slabs"
        elif solver == "surf":
            slab_sim_name = "integration_test_surf_slabs"
        else:
            raise ValueError(f"Unknown solver {solver}.")

        path_slab_sim_top = PH.directory_slab_simulation(slab_sim_name=slab_sim_name)

        # Remove the old test directory if it exists
        if os.path.exists(path_slab_sim_top):
            rmtree(path_slab_sim_top)

        p1 = PH.file_slab_target(
            slab_sim_name=slab_sim_name, signal_id=0, resampled=False
        )
        p2 = PH.file_slab_target(
            slab_sim_name=slab_sim_name, signal_id=1, resampled=False
        )
        p3 = PH.file_slab_target(
            slab_sim_name=slab_sim_name, signal_id=3, resampled=False
        )

        # Check that the target files do not exist before generating them. If they do, the test tells nothing.
        self.assertFalse(os.path.exists(p1))
        self.assertFalse(os.path.exists(p2))
        self.assertFalse(os.path.exists(p3))

        # Generate some random PROSPECT leaves to use for testing
        SMI.generate_prospect_leaf_random(slab_sim_name=slab_sim_name, leaf_count=2)
        SMI.generate_prospect_leaf(slab_sim_name=slab_sim_name, signal_id=3, w=0.001)

        # Check that the target files exist after generating them
        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertTrue(os.path.exists(p3))

        # Reduce the number of bands to four for quick testing
        new_sampling = [450, 500, 550, 1930]
        SMI.resample_slab_sim_target(slab_sim_name=slab_sim_name, wls=new_sampling)

        # Find resampled target files and check they exist
        p1 = PH.file_slab_target(
            slab_sim_name=slab_sim_name, signal_id=0, resampled=True
        )
        p2 = PH.file_slab_target(
            slab_sim_name=slab_sim_name, signal_id=1, resampled=True
        )
        p3 = PH.file_slab_target(
            slab_sim_name=slab_sim_name, signal_id=3, resampled=True
        )

        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertTrue(os.path.exists(p3))

        SMI.solve_slab_material_parameters(
            runtime=self.runtime,
            slab_sim_name=slab_sim_name,
            clear_old_results=True,
            solver=solver,
            copyof=None,
            solver_dirname=solver_dirname,
        )

        self.check_existance_of_signal_result_files(
            slab_sim_name=slab_sim_name, signal_ids=[0, 1, 3]
        )
        self.check_existence_of_common_files(slab_sim_name=slab_sim_name)

        # We could check also the file contents, but if the program is so broken that the content
        # is not correct, we are in trouble anyway.

    def run_advanced_solvers(self, solver_dirname: str = None):
        """Tests that the slab simulation run with the advanced solvers works as expected.

        The advanced solvers are the neural network and surface fitting solvers.
        """

        self.check_nn_and_surf_result_files(solver="nn", solver_dirname=solver_dirname)
        self.check_nn_and_surf_result_files(
            solver="surf", solver_dirname=solver_dirname
        )

    def run_slab_optimization(self, solver_dirname: str = None):
        """Tests that the slab simulation run with the optimization solver works as expected.

        The optimizer is run mainly when training the nn model, so it is important to be tested.
        """

        slab_sim_opt_name = "integration_test_opt_slabs"
        path_slab_sim_top = PH.directory_slab_simulation(
            slab_sim_name=slab_sim_opt_name
        )

        # Remove the old test directory if it exists
        if os.path.exists(path_slab_sim_top):
            rmtree(path_slab_sim_top)

        p1 = PH.file_slab_target(
            slab_sim_name=slab_sim_opt_name, signal_id=0, resampled=False
        )

        self.assertFalse(os.path.exists(p1))
        SMI.generate_prospect_leaf_random(slab_sim_name=slab_sim_opt_name, leaf_count=1)
        self.assertTrue(os.path.exists(p1))

        # Reduce the number of channels to four for quick testing
        new_sampling = [550, 650]
        SMI.resample_slab_sim_target(slab_sim_name=slab_sim_opt_name, wls=new_sampling)

        SMI.solve_slab_material_parameters(
            runtime=self.runtime,
            slab_sim_name=slab_sim_opt_name,
            clear_old_results=True,
            solver="opt",
            copyof=None,
            solver_dirname=solver_dirname,
        )

        # Check that for all wavelengths, there exists a subresult file. This check is done only for the optimizer.
        signal_id = 0
        for wl in new_sampling:
            p = PH.file_wl_result(
                slab_sim_name=slab_sim_opt_name, signal_id=signal_id, wl=wl
            )
            with self.subTest(p=p):
                error_msg = (
                    f"Result file for signal {signal_id} and wavelength {wl} does not exist "
                    f"at path {p}."
                )
                self.assertTrue(os.path.exists(p), msg=error_msg)

        self.check_existance_of_signal_result_files(
            slab_sim_name=slab_sim_opt_name, signal_ids=[0]
        )
        self.check_existence_of_common_files(slab_sim_name=slab_sim_opt_name)

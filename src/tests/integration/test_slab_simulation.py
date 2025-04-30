

import os
import unittest
from shutil import rmtree
from unittest import TestCase

from src.slab_model import interface as SMI
from src.data import path_handling as PH, toml_handling as TH


class Test(TestCase):

    def check_existence_of_common_files(self, slab_sim_name):
        """ Check that the common files exist in the slab simulation directory after the simulation is run. """

        print(f"Checking existence of common files in {slab_sim_name} directory.")

        path_slab_sim_result = PH.path_file_slab_sim_result(slab_sim_name=slab_sim_name)
        fail_msg = f"Could not find the slab simulation result file at {path_slab_sim_result}."
        self.assertTrue(os.path.exists(path_slab_sim_result), msg=fail_msg)
        print(f"Found slab simulation result file at {path_slab_sim_result}.")



    def test_generate_prospect_leaf(self):

        slab_sim_name = "integration_test_slabs"
        path_slab_sim_top = PH.path_directory_slab_simulation(slab_sim_name=slab_sim_name)

        # Remove the old test directory if it exists
        if os.path.exists(path_slab_sim_top):
            rmtree(path_slab_sim_top)

        p1 = PH.path_file_target(set_name=slab_sim_name, sample_id=0, resampled=False)
        p2 = PH.path_file_target(set_name=slab_sim_name, sample_id=1, resampled=False)
        p3 = PH.path_file_target(set_name=slab_sim_name, sample_id=3, resampled=False)

        self.assertFalse(os.path.exists(p1))
        self.assertFalse(os.path.exists(p2))
        self.assertFalse(os.path.exists(p3))

        """ Generate some random PROSPECT leaves to use for testing."""
        SMI.generate_prospect_leaf_random(set_name=slab_sim_name, leaf_count=2)
        SMI.generate_prospect_leaf(set_name=slab_sim_name, sample_id=3, w=0.001)

        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertTrue(os.path.exists(p3))

        # Reduce the number of channels to four for quick testing
        new_sampling = [450,500,550,1930]
        SMI.resample_leaf_targets(set_name=slab_sim_name, new_sampling=new_sampling) # resample leaf spectra

        # Find resampled target files and check they exist
        p1 = PH.path_file_target(set_name=slab_sim_name, sample_id=0, resampled=True)
        p2 = PH.path_file_target(set_name=slab_sim_name, sample_id=1, resampled=True)
        p3 = PH.path_file_target(set_name=slab_sim_name, sample_id=3, resampled=True)

        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertTrue(os.path.exists(p3))

        SMI.solve_leaf_material_parameters(
            set_name=slab_sim_name, clear_old_results=True,
            resolution = None, use_dumb_sampling = False, solver = 'nn',
            copyof = None, plot_resampling = False)

        # Check that for all signals and all wavelengths, there exists a result file
        for signal_id in [0, 1, 3]:
            p = PH.path_file_signal_result(slab_sim_name=slab_sim_name, signal_id=signal_id)
            with self.subTest(p=p):
                error_msg = f"Result file for signal {signal_id} does not exist at path {p}."
                self.assertTrue(os.path.exists(p), msg=error_msg)

        self.check_existence_of_common_files(slab_sim_name=slab_sim_name)

        # We could check also the file contents, but if the program is so broken that the content
        # is not correct, we are in trouble anyway.

    def test_slab_optimization(self):
        """ Tests that the slab simulation run with the optimization solver works as expected.

        The optimizer is run mainly when training the nn model, so it is important to be tested.
        """

        slab_sim_opt_name = "integration_test_opt_slabs"
        path_slab_sim_top = PH.path_directory_slab_simulation(slab_sim_name=slab_sim_opt_name)

        # Remove the old test directory if it exists
        if os.path.exists(path_slab_sim_top):
            rmtree(path_slab_sim_top)

        p1 = PH.path_file_target(set_name=slab_sim_opt_name, sample_id=0, resampled=False)

        self.assertFalse(os.path.exists(p1))
        SMI.generate_prospect_leaf_random(set_name=slab_sim_opt_name, leaf_count=1)
        self.assertTrue(os.path.exists(p1))

        # Reduce the number of channels to four for quick testing
        new_sampling = [550,650]
        SMI.resample_leaf_targets(set_name=slab_sim_opt_name, new_sampling=new_sampling)

        SMI.solve_leaf_material_parameters(
            set_name=slab_sim_opt_name, clear_old_results=True,
            resolution=None, use_dumb_sampling=False, solver='opt',
            copyof=None, plot_resampling=False)

        # Check that for all wavelengths, there exists a subresult file
        signal_id = 0
        for wl in new_sampling:
            p = PH.path_file_wl_result(set_name=slab_sim_opt_name, sample_id=signal_id, wl=wl)
            with self.subTest(p=p):
                error_msg = (f"Result file for signal {signal_id} and wavelength {wl} does not exist "
                             f"at path {p}.")
                self.assertTrue(os.path.exists(p), msg=error_msg)

        # Check that signal result file exists
        p_signal_result = PH.path_file_signal_result(slab_sim_name=slab_sim_opt_name, signal_id=signal_id)
        self.assertTrue(os.path.exists(p_signal_result))

        self.check_existence_of_common_files(slab_sim_name=slab_sim_opt_name)

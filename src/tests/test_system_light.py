import os
from shutil import rmtree
import unittest  # needed for skipping tests
from unittest import TestCase
import logging
import numpy as np

from src.system_simulation import lighting
from src.setup import initialization
from src.data import path_handling as PH, file_handling as FH


# @unittest.skip("Skipping for now")
class TestSystemLight(TestCase):
    """Tests system simulation light loading functionality.

    Cleans up the test light file after the test.
    """

    def setUp(self) -> None:
        self.runtime = initialization.initialize()

    def tearDown(self) -> None:
        logging.shutdown()

    def test_load_light(self) -> None:
        """Test loading light data."""

        logging.info("Test loading default sun light")
        wls, irradiances = lighting.load_light()
        self.assertIsInstance(wls, np.ndarray)
        self.assertIsInstance(irradiances, np.ndarray)
        self.assertGreater(len(wls), 0)
        self.assertGreater(len(irradiances), 0)

        logging.info("Test loading default sky light")
        wls_sky, irradiances_sky = lighting.load_light(lighting_type="sky")
        self.assertIsInstance(wls_sky, np.ndarray)
        self.assertIsInstance(irradiances_sky, np.ndarray)
        self.assertGreater(len(wls_sky), 0)
        self.assertGreater(len(irradiances_sky), 0)

        logging.info("Test loading a custom light file")
        custom_file = "test_light_spectra.txt"
        path_light_spectra = PH.directory_light_spectra()
        path_light_spectra = PH.join(path_light_spectra, custom_file)
        write_test_light_file(path_light_spectra=path_light_spectra)

        # This is written to where the default light files are stored.
        wls_custom, irradiances_custom = lighting.load_light(file_name=custom_file)
        self.assertEqual(len(wls_custom), 3)
        self.assertEqual(len(irradiances_custom), 3)

        # Clean up the test file
        if os.path.exists(path_light_spectra):
            os.remove(path_light_spectra)

        logging.info("Test loading a custom light file in system simulation context")

        system_sim_name = "test_system_light"
        FH.create_top_level_system_sim_directories(system_sim_name=system_sim_name)
        path_system_sim = PH.directory_system_simulation(
            system_sim_name=system_sim_name
        )
        path_system_sim = PH.join(path_system_sim, custom_file)
        write_test_light_file(path_light_spectra=path_system_sim)
        wls_custom, irradiances_custom = lighting.load_light(
            file_name=custom_file, system_sim_name=system_sim_name
        )
        self.assertEqual(len(wls_custom), 3)
        self.assertEqual(len(irradiances_custom), 3)

        # Clean up the system simulation directory
        p_system_sim_dir = PH.directory_system_simulation(
            system_sim_name=system_sim_name
        )
        if os.path.exists(p_system_sim_dir):
            rmtree(p_system_sim_dir)


def write_test_light_file(path_light_spectra):
    """Write a test light file with 1 nm resolution for loading."""

    with open(path_light_spectra, "w") as f:
        f.write("# Wavelength (nm) Irradiance\n")
        f.write("400 0.1\n")
        f.write("401 0.2\n")
        f.write("402 0.3\n")

"""

This module test the system simulation functionality.

"""

import os
import unittest  # needed for skipping tests
from shutil import rmtree
from unittest import TestCase
import logging
import numpy as np

from src.setup import initialization
from src.slab_model import interface as SMI
from src.data import path_handling as PH, toml_handling as TH
from src.data import cube_handling as CH
from src.forest import forest
from src.rendering import blender_control as BC
from src.setup.runtime_environment import RuntimeEnvironment
from src.blender_scripts import forest_control as FCtrl


class TestSystemSimulation(TestCase):

    def setUp(self) -> None:
        self.runtime = initialization.initialize()

    def tearDown(self) -> None:
        logging.shutdown()

    def test_forest_pipe(self) -> None:
        """This is a testing box for all forest canopy simulation functionality."""

        logging.info(f"Running system simulation test")

        rng = np.random.default_rng(333)
        slab_sim_name = "slabs_for_system_test"
        system_sim_name_master = "master_system_for_test"
        system_sim_name_slave = "slave_system_for_test"

        # Use pre-calculated soil spectra and default sun and sky spectra.
        #   They are automatically interpolated to match the leaf spectra bands.
        soil_name = "median_humid_clay_reflectance"
        sun_name = "default_sun"
        sky_name = "default_sky"

        slab_material_names = ["Leaf material 1", "Leaf material 2", "Leaf material 3"]

        # Generating low resolution random leaves.
        # First two random leaves
        # SMI.generate_prospect_leaf_random(set_name=slab_sim_name, leaf_count=2)
        # # THen one with low water content
        # SMI.generate_prospect_leaf(set_name=slab_sim_name, sample_id=3, w=0.001)
        # # Resample to include only a few bands so the test runs in reasonable time
        # new_sampling = [450, 550, 650, 1930]
        # SMI.resample_leaf_targets(set_name=slab_sim_name, new_sampling=new_sampling)
        # SMI.solve_leaf_material_parameters(
        #     set_name=slab_sim_name, clear_old_results=True, runtime=self.runtime
        # )

        # Here we create a new forest scene from the template. Should be uncommented for the first run.
        # This creates a new "master" forest you can use to generate other similar forests later.

        # Pack leaf data for forest scene initialization.
        leaves = [
            (slab_sim_name, 0, slab_material_names[0]),
            (slab_sim_name, 1, slab_material_names[1]),
            (slab_sim_name, 3, slab_material_names[2]),
        ]

        forest.init(
            leaves=leaves,
            conf_type="m2m",
            rng=rng,
            custom_forest_id=system_sim_name_master,
            soil_name=soil_name,
            sun_file_name=sun_name,
            sky_file_name=sky_name,
        )

        self.check_system_simulation_file_existence(
            system_sim_name=system_sim_name_master,
            slab_material_names=slab_material_names,
        )

        # Setup master and render preview
        BC.setup_forest(
            forest_id=system_sim_name_master,
            leaf_name_list=slab_material_names,
            runtime=self.runtime,
        )

        # BC.render_forest(
        #     forest_id=system_sim_name_master,
        #     render_mode="preview",
        #     runtime=self.runtime,
        # )

        # TODO check the existance of preview renders

        forest.init(
            leaves=leaves,
            conf_type="m2s",
            rng=rng,
            custom_forest_id=system_sim_name_slave,
            copy_forest_id=system_sim_name_master,
            soil_name=soil_name,
            sun_file_name=sun_name,
            sky_file_name=sky_name,
        )

        # Check that the files were copied correctly
        self.check_system_simulation_file_existence(
            system_sim_name=system_sim_name_slave,
            slab_material_names=slab_material_names,
        )

        # Read the master forest control file and modify it and write it to the slave
        system_control = FCtrl.read_forest_control(forest_id=system_sim_name_master)
        key_forest = "Forest"
        key_min_tree_separation = "Minimum tree separation [m]"
        old_separation = system_control[key_forest][key_min_tree_separation]["Value"]
        mult = 0.5
        new_separation = old_separation * mult
        system_control[key_forest][key_min_tree_separation]["Value"] = new_separation
        # print(system_control)

        FCtrl.write_forest_control(
            forest_id=system_sim_name_slave,
            control_dict=system_control,
            global_master=False,
        )

        # Check that the new separation is written to the slave control file
        system_control_slave = FCtrl.read_forest_control(
            forest_id=system_sim_name_slave
        )
        slave_separation = system_control_slave[key_forest][key_min_tree_separation][
            "Value"
        ]
        self.assertAlmostEqual(new_separation, slave_separation)

        # Running forest.init only copies files. Running setup makes the
        #   Blender scene renderable. This also applies the changes in the control file.
        BC.setup_forest(
            forest_id=system_sim_name_slave,
            leaf_name_list=slab_material_names,
            runtime=self.runtime,
        )

        # Render bands for spectral cube along with additional images
        # BC.render_forest(
        #     forest_id=system_sim_name_slave, render_mode="preview", runtime=self.runtime
        # )
        # BC.render_forest(
        #     forest_id=system_sim_name_slave,
        #     render_mode="visibility",
        #     runtime=self.runtime,
        # )
        # BC.render_forest(
        #     forest_id=system_sim_name_slave,
        #     render_mode="spectral",
        #     runtime=self.runtime,
        # )

        # TODO check existence of the rest of the rendered files

        # Construct spectral cube in ENVI format
        # CH.construct_envi_cube(forest_id=system_sim_name_slave)
        # TODO check that the spectral cube is built

    def check_system_simulation_file_existence(
        self, system_sim_name: str, slab_material_names
    ):
        should_exist = [
            PH.path_file_system_simulation_blend(simulation_name=system_sim_name),
            PH.path_directory_system_simulation(forest_id=system_sim_name),
            PH.path_file_system_slab_csv(
                forest_id=system_sim_name, leaf_index=slab_material_names[0]
            ),
            PH.path_file_system_slab_csv(
                forest_id=system_sim_name, leaf_index=slab_material_names[0]
            ),
            PH.path_file_system_slab_csv(
                forest_id=system_sim_name, leaf_index=slab_material_names[0]
            ),
            PH.path_file_system_forest_sun_spectra_csv(forest_id=system_sim_name),
            PH.path_file_forest_sky_csv(forest_id=system_sim_name),
        ]

        for path in should_exist:
            with self.subTest(path=path):
                error_msg = f"Could not find anything from path '{path}'."
                self.assertTrue(os.path.exists(path), msg=error_msg)
                logging.info(f"Found path '{path}'")

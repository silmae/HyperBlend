"""

This module test the system simulation functionality.

"""

import os
import unittest  # needed for skipping tests
from unittest import TestCase
import logging
import numpy as np

from src.setup import initialization
from src.slab_model import interface as SMI
from src.data import path_handling as PH
from src.data import cube_handling as CH
from src.system_simulation import forest
from src.rendering import blender_control as BC
from src.blender_scripts import forest_control as FCtrl
from src import constants as C


# @unittest.skip("Skipping for now")
class TestSystemSimulation(TestCase):

    def setUp(self) -> None:
        self.runtime = initialization.initialize()

    def tearDown(self) -> None:
        logging.shutdown()

    def test_forest_pipe(self) -> None:
        """This is a testing box for all system_simulation canopy simulation functionality."""

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
        SMI.generate_prospect_leaf_random(set_name=slab_sim_name, leaf_count=2)
        # THen one with low water content
        SMI.generate_prospect_leaf(set_name=slab_sim_name, sample_id=3, w=0.001)
        # Resample to include only a few bands so the test runs in reasonable time
        new_sampling = [450, 550, 650, 1930]
        SMI.resample_leaf_targets(set_name=slab_sim_name, new_sampling=new_sampling)
        SMI.solve_leaf_material_parameters(
            slab_sim_name=slab_sim_name, clear_old_results=True, runtime=self.runtime
        )

        # Pack leaf data for system_simulation scene initialization.
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

        BC.render_forest(
            forest_id=system_sim_name_master,
            render_mode="preview",
            runtime=self.runtime,
        )

        # Check that previews were rendered
        image_path = PH.file_system_sim_preview(
            system_sim_name=system_sim_name_master,
            image_name=C.filename_system_sim_preview_sleeper,
        )
        self.assertTrue(os.path.exists(image_path))

        image_path = PH.file_system_sim_preview(
            system_sim_name=system_sim_name_master,
            image_name=C.filename_system_sim_preview_drone,
        )
        self.assertTrue(os.path.exists(image_path))

        image_path = PH.file_system_sim_preview(
            system_sim_name=system_sim_name_master,
            image_name=C.filename_system_sim_preview_walker,
        )
        self.assertTrue(os.path.exists(image_path))

        image_path = PH.file_system_sim_preview(
            system_sim_name=system_sim_name_master,
            image_name=C.filename_system_sim_preview_trees,
        )
        self.assertTrue(os.path.exists(image_path))

        # Initialize a new slave system simulation from the master
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

        # Read the master system_simulation control file and modify it and write it to the slave
        #   Setting the minimum tree separation to lower value spawns more trees so the
        #   change will be visible in the preview images.
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

        # Running system_simulation.init only copies files. Running setup makes the
        #   Blender scene renderable. This also applies the changes in the control file.
        BC.setup_forest(
            forest_id=system_sim_name_slave,
            leaf_name_list=slab_material_names,
            runtime=self.runtime,
        )

        # Render spectral cube, visibility maps, and previews
        BC.render_forest(
            forest_id=system_sim_name_slave, render_mode="preview", runtime=self.runtime
        )
        BC.render_forest(
            forest_id=system_sim_name_slave,
            render_mode="visibility",
            runtime=self.runtime,
        )
        BC.render_forest(
            forest_id=system_sim_name_slave,
            render_mode="spectral",
            runtime=self.runtime,
        )

        for i, wl in enumerate(new_sampling):
            p = PH.join(
                PH.directory_system_rend_spectral(
                    system_sim_name=system_sim_name_slave
                ),
                f"band_000{i+1}.tiff",
            )
            with self.subTest(path=p):
                error_msg = f"Could not find rendered image from path '{p}'."
                self.assertTrue(os.path.exists(p), msg=error_msg)
                logging.info(f"Found rendered image at '{p}'")

        ref_map_list = PH.list_reference_visibility_maps(
            system_sim_name=system_sim_name_slave
        )
        self.assertGreater(len(ref_map_list), 0)
        for map in ref_map_list:
            map_path = PH.file_visibility_map(
                system_sim_name=system_sim_name_slave, file_name=map
            )
            with self.subTest(path=map_path):
                error_msg = f"Could not find visibility map from path '{map_path}'."
                self.assertTrue(os.path.exists(map_path), msg=error_msg)
                logging.info(f"Found visibility map at '{map_path}'")

        # Construct spectral cube in ENVI format
        CH.construct_envi_cube(system_sim_name=system_sim_name_slave)

        p = PH.directory_system_spectral_cube(system_sim_name=system_sim_name_slave)
        self.assertTrue(
            os.path.exists(p), msg=f"Spectral cube could not be found from '{p}'."
        )

        logging.info("System simulation test finished.")

    def check_system_simulation_file_existence(
        self, system_sim_name: str, slab_material_names
    ):
        should_exist = [
            PH.file_blend_system_simulation(simulation_name=system_sim_name),
            PH.directory_system_simulation(system_sim_name=system_sim_name),
            PH.file_system_slab_csv(
                system_sim_name=system_sim_name,
                slab_material_name=slab_material_names[0],
            ),
            PH.file_system_slab_csv(
                system_sim_name=system_sim_name,
                slab_material_name=slab_material_names[0],
            ),
            PH.file_system_slab_csv(
                system_sim_name=system_sim_name,
                slab_material_name=slab_material_names[0],
            ),
            PH.file_system_sim_light_spectra_csv(
                system_sim_name=system_sim_name,
                light_file_name=C.file_blender_default_sun,
            ),
            PH.file_system_sim_light_spectra_csv(
                system_sim_name=system_sim_name,
                light_file_name=C.file_blender_default_sky,
            ),
        ]

        for path in should_exist:
            with self.subTest(path=path):
                error_msg = f"Could not find anything from path '{path}'."
                self.assertTrue(os.path.exists(path), msg=error_msg)
                logging.info(f"Found path '{path}'")

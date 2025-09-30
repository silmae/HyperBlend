"""

This module is used to generate leaves for the dataset paper

"""

import logging
import os.path
import pandas as pd

import numpy as np
import shutil
from matplotlib import pyplot as plt

from setup.runtime_environment import RuntimeEnvironment
from src.slab_model import interface as SI
from src.system_simulation import forest
from rendering import blender_control as BC
from src.data import (
    cube_handling as CH,
    path_handling as PH,
    file_handling as FH,
    toml_handling as TH,
)
from src.utils import data_utils as DU


slab_sim_names = ["Manitoba Maple", "American Elm", "Crab apple"]
slab_sim_name_pr = "dataset_paper_prospect_leaves"


def run(runtime: RuntimeEnvironment):
    """Just a little run function to be called from main to keep it neat."""

    # logging.info("Dataset run started.")

    # generate_leaves()
    # solve_leaves(runtime=runtime, slab_sim_names=slab_sim_names)

    # generate_random_prospect_leaves(slab_sim_name=slab_sim_name_pr, leaf_count=5)
    # solve_leaves(runtime=runtime, slab_sim_names=[slab_sim_name_pr])

    # rng = np.random.default_rng(1243567)
    # generate_forest_master(runtime=runtime, rng=rng)
    # lotus_to_hb(runtime)
    miu(runtime=runtime)


def miu(runtime: RuntimeEnvironment):

    lotus_species_names = [
        "LOTUS Saskatoon berry",
        "LOTUS Oak",
        "LOTUS Elm",
        "LOTUS Mountain ash",
        "LOTUS Green ash",
        "LOTUS American elm",
        "LOTUS Grape",
        "LOTUS Purple cherry",
        "LOTUS Manitoba maple",
    ]

    # In case you forgot to resample them earlier, they have to be solved again.
    # Just leaving this snippet for future reference.
    # for slab_sim_name in lotus_species_names:
    #     SI.resample_slab_sim_target(
    #         slab_sim_name=slab_sim_name, range_start=400, range_end=2500, resolution=5
    #     )  # resample leaf spectra
    #
    #     SI.solve_leaf_material_parameters(
    #         runtime=runtime,
    #         slab_sim_name=slab_sim_name,
    #         clear_old_results=True,
    #         range_start=400,
    #         range_end=2500,
    #         resolution=5,
    #         solver_dirname="Iterative slab",
    #     )  # run slab simulation

    # Some ID's and names. Can be uncommented all times
    # Scene IDs
    forest_id_master = "dataset_forest_TEST"

    # Use pre-calculated soil spectra and default sun and sky spectra. They are automatically
    # interpolated to match the leaf spectra bands. Can be uncommented all times
    soil_name = "wet_peat_reflectance"
    sun_name = "default_sun"
    sky_name = "default_sky"

    # Here we create a new system_simulation scene from the template. Should be uncommented for the first run.
    # This creates a new "master" system_simulation you can use to generate other similar forests later.

    # Pack leaf data for system_simulation scene initialization. This can be uncommented all times
    leaves = [
        (lotus_species_names[4], 2, "Slab material 1"),
        (lotus_species_names[4], 4, "Slab material 2"),
        (lotus_species_names[5], 0, "Slab material 3"),
        (lotus_species_names[5], 2, "Slab material 4"),
        (lotus_species_names[5], 0, "Slab material 5"),
        (lotus_species_names[5], 2, "Slab material 6"),
    ]

    def run_next_resolution(
        scene_id: int,
        resolution: int,
        do_copy=False,
        run_setup_and_render: bool = False,
    ):

        current_level_name = f"dataset_{resolution}_{scene_id}"

        if do_copy:
            forest.init(
                leaves=leaves,
                conf_type="m2m",
                copy_forest_id=high_level_name,
                custom_forest_id=current_level_name,
                soil_name=soil_name,
                sun_file_name=sun_name,
                sky_file_name=sky_name,
            )

        if run_setup_and_render:
            BC.setup_system_sim_scene(
                runtime=runtime,
                system_sim_name=current_level_name,
                leaf_name_list=[
                    "Slab material 1",
                    "Slab material 2",
                    "Slab material 3",
                    "Slab material 4",
                    "Slab material 5",
                    "Slab material 6",
                ],
            )

            BC.render_forest(
                runtime=runtime,
                system_sim_name=current_level_name,
                render_mode="preview",
            )

            # Visibility maps only for the high res cube. The rest are calculated manually.
            if resolution >= 1024:
                BC.render_forest(
                    runtime=runtime,
                    system_sim_name=current_level_name,
                    render_mode="visibility",
                )

            BC.render_forest(
                runtime=runtime,
                system_sim_name=current_level_name,
                render_mode="spectral",
            )

            # Construct spectral cube in ENVI format
            CH.construct_envi_cube(
                system_sim_name=current_level_name,
                system_sim_name_for_white_signal=high_level_name,
            )

    # This is the master master that is used to spawn the highest resolution forests
    # forest.init(
    #     leaves=leaves,
    #     conf_type="m2m",
    #     custom_forest_id=forest_id_master,
    #     soil_name=soil_name,
    #     sun_file_name=sun_name,
    #     sky_file_name=sky_name,
    # )

    # BC.generate_forest_control(
    #     runtime=runtime, system_sim_name=forest_id_master, global_master=False
    # )

    # # Setup master and render preview
    # BC.setup_system_sim_scene(
    #     runtime=runtime,
    #     system_sim_name=forest_id_master,
    #     leaf_name_list=[
    #         "Slab material 1",
    #         "Slab material 2",
    #         "Slab material 3",
    #         "Slab material 4",
    #     ],
    # )

    high_level_name = "dataset_1024_1"

    # run_next_resolution(
    #     scene_id=1, resolution=256, do_copy=False, run_setup_and_render=True
    # )
    run_next_resolution(
        scene_id=1, resolution=64, do_copy=False, run_setup_and_render=True
    )
    run_next_resolution(
        scene_id=1, resolution=16, do_copy=False, run_setup_and_render=True
    )
    run_next_resolution(
        scene_id=1, resolution=4, do_copy=False, run_setup_and_render=True
    )

    # forest.init(
    #     leaves=leaves,
    #     conf_type="m2m",
    #     copy_forest_id=forest_id_master,
    #     custom_forest_id=high_level_name,
    #     soil_name=soil_name,
    #     sun_file_name=sun_name,
    #     sky_file_name=sky_name,
    # )
    #
    # # # Setup the high level forest scene
    # BC.setup_system_sim_scene(
    #     runtime=runtime,
    #     system_sim_name=high_level_name,
    #     leaf_name_list=[
    #         "Slab material 1",
    #         "Slab material 2",
    #         "Slab material 3",
    #         "Slab material 4",
    #         "Slab material 5",
    #         "Slab material 6",
    #     ],
    # )

    # BC.render_forest(
    #     runtime=runtime, system_sim_name=high_level_name, render_mode="preview"
    # )
    # BC.render_forest(
    #     runtime=runtime, system_sim_name=high_level_name, render_mode="visibility"
    # )
    # BC.render_forest(
    #     runtime=runtime, system_sim_name=high_level_name, render_mode="spectral"
    # )

    # Construct spectral cube in ENVI format
    # CH.construct_envi_cube(system_sim_name=high_level_name)


def lotus_to_hb(runtime: RuntimeEnvironment):

    # Assume lotus main directory is two levels higher than current working directory
    lotus_main_dir = os.path.abspath("../../FRDR_dataset/LOTUS")
    if os.path.exists(lotus_main_dir):
        logging.info(f"Lotus data found at {lotus_main_dir}.")
    cary_dir = PH.join(lotus_main_dir, "Hemispherical Data", "Cary 5000")
    if os.path.exists(cary_dir):
        logging.info(f"Found Cary 5000 data at {cary_dir}.")
    image_dir = PH.join(lotus_main_dir, "Images of leaves")
    if os.path.exists(image_dir):
        logging.info(f"Found images at {image_dir}.")
    meta_excel_path = PH.join(lotus_main_dir, "Leaf Information.xlsx")
    if os.path.exists(meta_excel_path):
        logging.info(f"Found metadata at {meta_excel_path}.")

    cols = "B:V"
    # Read the metadata Excel file
    meta_df = pd.read_excel(meta_excel_path, sheet_name=1, index_col=1, usecols=cols)
    # print(meta_df)

    common_names = meta_df["Common Name"].unique()

    select_by_common_name = [
        "Saskatoon berry",
        "Oak",
        "Elm",
        "Mountain ash",
        "Green ash",
        "American elm",
        "Grape",
        "Purple cherry",
        "Manitoba maple",
        "Crab apple",
    ]

    slab_sim_names = []

    for index, row in meta_df.iterrows():

        # each row is returned as a pandas series
        common_name = row["Common Name"]
        if common_name in select_by_common_name:

            file_base_name = row["FileName"]

            # Skip abaxial sides of the leaves
            if file_base_name.endswith("b"):
                continue

            print(f"Processing {file_base_name} with common name {common_name}")

            refl_file_name = file_base_name + "_R.txt"
            tran_file_name = file_base_name + "_T.txt"
            relf_file_path = PH.join(cary_dir, refl_file_name)
            tran_file_path = PH.join(cary_dir, tran_file_name)

            # Load csv files into numpy arrays
            refl = np.loadtxt(relf_file_path)
            tran = np.loadtxt(tran_file_path)

            # Skip wavelengths below 400 nm. The dataset starts from 200 nm
            wls = refl[200:, 0]
            refl = refl[200:, 1]
            tran = tran[200:, 1]

            # Debug plot
            # plt.plot(wls, refl, label=f"Reflectance {common_name}")
            # plt.plot(wls, tran, label=f"Transmittance {common_name}")
            # plt.show()
            # print(tran)

            slab_sim_name = "LOTUS " + common_name
            FH.create_top_level_slab_sim_directories(slab_sim_name=slab_sim_name)

            # Check existing target IDs and create a new one with ID one greater
            existing_target_ids = FH.list_target_ids(slab_sim_name=slab_sim_name)
            target_id = len(existing_target_ids)
            # print(f"Target ID: {target_id} for {slab_sim_name}")

            # They have to be actually created here, so that they can be found later
            FH.create_signal_optimization_directories(slab_sim_name, target_id)
            target_data = DU.pack_target(wls, refl, tran)
            TH.write_target(
                slab_sim_name=slab_sim_name, data=target_data, signal_id=target_id
            )

            # Then copy leaf images and metadata
            src_image_path = PH.join(image_dir, file_base_name + ".JPG")
            dst_image_path = PH.join(
                PH.directory_slab_simulation(slab_sim_name), file_base_name + ".JPG"
            )
            shutil.copy(src_image_path, dst_image_path)

            meta_file_name = file_base_name + ".toml"
            meta_dict = row.to_dict()
            TH.write_dict_as_toml(
                meta_dict, PH.directory_slab_simulation(slab_sim_name), meta_file_name
            )
            slab_sim_names.append(slab_sim_name)

    # Finally, solve material parameters
    for slab_sim_name in slab_sim_names:
        SI.solve_leaf_material_parameters(
            runtime=runtime,
            slab_sim_name=slab_sim_name,
            solver="nn",
            solver_dirname="Iterative slab",
            range_start=400,
            range_end=2500,
            resolution=5,
        )


def generate_random_prospect_leaves(slab_sim_name: str, leaf_count: int = 2):

    SI.generate_prospect_leaf_random(slab_sim_name=slab_sim_name, leaf_count=leaf_count)


def generate_leaves():
    """Generate tree leaves for the dataset.

    The leaf parameters are based on paper

    A new dataset of leaf optical traits to include biophysical parameters in
    addition to spectral and biochemical assessment
    https://doi.org/10.1016/j.rse.2024.114424

    Leaves generated here are::

        1. Crab apple (Malus sp. Mill.) coded as APLE1
        2. Manitoba Maple (Acer negundo L.) coded CCAN2
        3. American Elm (Ulmus americana L.) coded AELM1

    .. note::
        Check EWT conversion from :math:`g / m^2` to cm in PROSPECT.
        And, how to set PROSPECT N parameter.

    TODO: add variations of leaves?
    """

    SI.generate_prospect_leaf(
        set_name=slab_sim_names[0],
        sample_id=0,
        n=None,
        ab=3.63,
        ar=3.27,
        brown=None,
        w=0.0065,
        m=0.0041,
        ant=7.34,
    )

    SI.generate_prospect_leaf(
        set_name=slab_sim_names[1],
        sample_id=0,
        n=None,
        ab=3.31,
        ar=0.41,
        brown=None,
        w=0.0090,
        m=0.0018,
        ant=0.50,
    )

    SI.generate_prospect_leaf(
        set_name=slab_sim_names[2],
        sample_id=0,
        n=None,
        ab=34.25,
        ar=5.65,
        brown=None,
        w=0.0084,
        m=0.0038,
        ant=0.38,
    )


def solve_leaves(runtime: RuntimeEnvironment, sims_to_solve_list):
    """Solve leaf parameters."""

    for slab_sim_name in sims_to_solve_list:
        SI.solve_leaf_material_parameters(
            runtime=runtime,
            slab_sim_name=slab_sim_name,
            resolution=5,
            range_start=400,
            range_end=900,
            solver="nn",
            solver_dirname="Iterative slab",
        )


def generate_forest_master(runtime: RuntimeEnvironment, rng: np.random.Generator):

    slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]
    system_sim_name_master = "dataset_paper_master"
    system_sim_name_slave = "dataset_paper_slave"

    # Pack leaf data for system_simulation scene initialization.
    leaves = [
        (slab_sim_names[0], 0, slab_material_names[0]),
        (slab_sim_names[1], 0, slab_material_names[1]),
        (slab_sim_names[2], 0, slab_material_names[2]),
    ]

    # leaves = [
    #     (slab_sim_name_pr, 1, slab_material_names[0]),
    #     (slab_sim_name_pr, 2, slab_material_names[1]),
    #     (slab_sim_name_pr, 3, slab_material_names[2]),
    # ]

    # forest.init(
    #     leaves=leaves,
    #     conf_type="m2m",
    #     rng=rng,
    #     custom_forest_id=system_sim_name_master,
    #     soil_name="median_humid_clay_reflectance",
    # )

    # Setup master and render preview
    # BC.setup_system_sim_scene(
    #     system_sim_name=system_sim_name_master,
    #     leaf_name_list=slab_material_names,
    #     runtime=runtime,
    # )

    forest.init(
        leaves=leaves,
        conf_type="m2s",
        rng=rng,
        custom_forest_id=system_sim_name_slave,
        copy_forest_id=system_sim_name_master,
        soil_name="median_humid_clay_reflectance",
    )

    BC.setup_system_sim_scene(
        runtime=runtime,
        system_sim_name=system_sim_name_slave,
        leaf_name_list=slab_material_names,
    )

    BC.render_forest(
        runtime=runtime,
        system_sim_name=system_sim_name_slave,
        render_mode="preview",
        silent=False,
    )

    BC.render_forest(
        runtime=runtime,
        system_sim_name=system_sim_name_slave,
        render_mode="spectral",
        silent=False,
    )

    BC.render_forest(
        runtime=runtime,
        system_sim_name=system_sim_name_slave,
        render_mode="visibility",
        silent=False,
    )

    CH.construct_envi_cube(system_sim_name=system_sim_name_slave)

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
from src.system_simulation import forest, lighting
from rendering import blender_control as BC
from src.data import (
    cube_handling as CH,
    path_handling as PH,
    file_handling as FH,
    toml_handling as TH,
)
from src.utils import data_utils as DU
from src.playground import create_dataset as CD


slab_sim_names = ["Manitoba Maple", "American Elm", "Crab apple"]
slab_sim_name_pr = "dataset_paper_prospect_leaves"

ancestor_scene = "dataset_ancestor"

# Define the LOTUS dataset sample names for fetching data
lotus_saskatoon_berry = {
    "slab_sim_name": "LOTUS Saskatoon berry",
    "lotus_codes": [
        "MARTN",
        "SMOKY",
    ],
    "common_name": "Saskatoon berry",
    "signal_count": 2,
}
lotus_oak = {
    "slab_sim_name": "LOTUS Oak",
    "lotus_codes": [
        "OAKDK",
        "OAKLT",
    ],
    "common_name": "Oak",
    "signal_count": 2,
}
lotus_elm = {
    "slab_sim_name": "LOTUS Elm",
    "lotus_codes": [
        "ELMDK",
    ],
    "common_name": "Elm",
    "signal_count": 1,
}
lotus_american_elm = {
    "slab_sim_name": "LOTUS American elm",
    "lotus_codes": [
        "AELM1",
        "AELM2",
        "AELM3",
        "AELM4",
    ],
    "common_name": "American elm",
    "signal_count": 4,
}
lotus_mountain_ash = {
    "slab_sim_name": "LOTUS Mountain ash",
    "lotus_codes": [
        "ASHDK",
    ],
    "common_name": "Mountain ash",
    "signal_count": 1,
}
lotus_green_ash = {
    "slab_sim_name": "LOTUS Green ash",
    "lotus_codes": [
        "GASH1",
        "GASH2",
        "GASH3",
        "GASH4",
        "GASH5",
    ],
    "common_name": "Green ash",
    "signal_count": 5,
}
lotus_grape = {
    "slab_sim_name": "LOTUS Grape",
    "lotus_codes": [
        "YGRPE",
    ],
    "common_name": "Grape",
    "signal_count": 1,
}
lotus_purple_cherry = {
    "slab_sim_name": "LOTUS Purple cherry",
    "lotus_codes": [
        "PCHRA",
        "PCHRB",
        "PCHRC",
        "PCHRD",
        "PCHRE",
        "PCHRF",
        "PCHRG",
        "PCHRH",
        "PCHRI",
    ],
    "common_name": "Purple cherry",
    "signal_count": 9,
}
lotus_manitoba_maple = {
    "slab_sim_name": "LOTUS Manitoba maple",
    "lotus_codes": [
        "CCAN1",
        "CCAN2",
    ],
    "common_name": "Manitoba maple",
    "signal_count": 2,
}

# Collect all LOTUS dicts for looping
lotus_sample_dicts = [
    lotus_saskatoon_berry,
    lotus_oak,
    lotus_elm,
    lotus_american_elm,
    lotus_mountain_ash,
    lotus_green_ash,
    lotus_grape,
    lotus_purple_cherry,
    lotus_manitoba_maple,
]


FWP1 = {
    "theme": "FWP1",
    "signals": [
        (lotus_elm, [0]),
        (lotus_manitoba_maple, [0, 1]),
        (lotus_green_ash, [0, 2, 4]),
    ],
}
FWP2 = {
    "theme": "FWP2",
    "signals": [
        (lotus_oak, [0, 1]),
        (lotus_saskatoon_berry, [0, 1]),
        (lotus_american_elm, [0, 1, 2]),
    ],
}
IWP1 = {
    "theme": "IWP1",
    "signals": [
        (lotus_oak, [0, 1]),
        (lotus_manitoba_maple, [0, 1]),
        (lotus_american_elm, [2, 3]),
        (lotus_mountain_ash, [0]),
    ],
}
OWP1 = {
    "theme": "OWP1",
    "signals": [
        (lotus_purple_cherry, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
}
OWP2 = {
    "theme": "OWP2",
    "signals": [(lotus_grape, [0])],
}

scenes_and_signals = [
    FWP1,  # general forest with wet peat
    FWP2,  # general forest with wet peat
    IWP1,  # inclined forest with wet peat
    OWP1,  # grape orchard with wet peat
    OWP2,  # cherry orchard with wet peat
]


def separate_spectral_renders(delete_originals=False, do_copy=True):
    """Rendered spectral bands that are used as a base to construct the spectral
    cubes are separated to a directory for smaller download size. They are
    needed only if one wants to reconstruct the spectral cubes again.
    """

    # Create a directory where to copy the spectral renders.
    path_root_dir_copy_to = PH.join(
        PH.directory_project_root(), "..", "Rendered spectral bands"
    )
    if not os.path.exists(path_root_dir_copy_to):
        os.makedirs(path_root_dir_copy_to)

    scene_number = 1
    for ss in scenes_and_signals:
        theme_wp = ss["theme"]  # soil: wet peat
        theme_ds = theme_wp.replace("WP", "DS")  # soil: dry sand
        for theme in [theme_wp, theme_ds]:
            for res in [1024, 256, 64, 16, 4]:
                sys_sim_name = f"{theme}_{res}"
                path_spectral_rend = PH.directory_system_rend_spectral(
                    system_sim_name=sys_sim_name
                )
                file_count = 0
                for file in os.listdir(path_spectral_rend):
                    file_count += 1
                    path_file_copy_from = PH.join(path_spectral_rend, file)
                    path_dir_copy_to = PH.join(
                        path_root_dir_copy_to, sys_sim_name, "rend", "Spectral"
                    )
                    if not os.path.exists(path_dir_copy_to):
                        os.makedirs(path_dir_copy_to)
                    path_file_copy_to = PH.join(path_dir_copy_to, file)

                    if do_copy:
                        shutil.copy(path_file_copy_from, path_file_copy_to)

                        print(
                            f"File copied from '{path_file_copy_from}' to '{path_dir_copy_to}'"
                        )

                    if delete_originals:
                        os.remove(path_file_copy_from)
                        print(f"Original file '{path_file_copy_from}' deleted.")

                    scene_number += 1

                print(
                    f"Scene {scene_number}: {sys_sim_name} has {file_count} spectral render files."
                )


def calculate_abundances():
    """Calculates abundances for each resolution level.

    Can be called once the dataset has been generated.
    """

    scene_number = 1
    for ss in scenes_and_signals:
        theme_wp = ss["theme"]  # soil: wet peat
        theme_ds = theme_wp.replace("WP", "DS")  # soil: dry sand
        for theme in [theme_wp, theme_ds]:

            # Get visibility maps from the full resolution simulation
            sys_sim_name_full_res = f"{theme}_1024"
            path_visibility = PH.directory_system_rend_visibility_maps(
                system_sim_name=sys_sim_name_full_res
            )

            # Debug print
            print(sys_sim_name_full_res)
            # print(path_visibility)

            if not os.path.exists(path_visibility):
                raise FileNotFoundError(
                    f"Visibility map directory {path_visibility} does not exist."
                )
            vismap_list = PH.list_visibility_maps(system_sim_name=sys_sim_name_full_res)
            # Exclude possible other files and rgb previews
            vismap_list = [
                file_name
                for file_name in vismap_list
                if (file_name.endswith(".tif") and not "rgb_preview" in file_name)
            ]
            # print(vsm_list)

            # Load leaf material name mapping to rename abundance maps
            leaf_mat_name_map = TH.read_toml_as_dict(
                directory=PH.directory_system_simulation(sys_sim_name_full_res),
                filename="leaf_material_map.toml",
            )
            list_slab_name = leaf_mat_name_map["slab_material_names"]
            list_lotus_code = leaf_mat_name_map["lotus_codes"]

            for res in [1024, 256, 64, 16, 4]:
                sys_sim_name_for_abund = f"{theme}_{res}"
                print(
                    f"Scene {scene_number}: Calculating abundances for {sys_sim_name_for_abund}..."
                )

                factor = 1
                if res == 256:
                    factor = 4
                elif res == 64:
                    factor = 16
                elif res == 16:
                    factor = 64
                elif res == 4:
                    factor = 256

                abundance_map_array = CD.ground_truth_abundances(
                    factor=factor,
                    visibility_maps_dir=path_visibility,
                    visibility_names=vismap_list,
                )

                # Create directory for abundance maps if it does not exist
                path_dir_abundances = PH.join(
                    PH.directory_system_simulation(sys_sim_name_for_abund),
                    "Abundance maps",
                )
                if not os.path.exists(path_dir_abundances):
                    os.makedirs(path_dir_abundances)

                # Save the full abundance map array as a single numpy file
                abundance_map_save_path = PH.join(
                    path_dir_abundances, f"abundances.npy"
                )
                np.save(abundance_map_save_path, abundance_map_array)

                map_name_indices = {
                    "note": "Mapping of abundance map indices to human readable names. "
                    "When you load the abundance map numpy array, you can find "
                    "a specific map by its index in this file and use the associated name."
                }

                # Rename material names to more human readable names and save visualizations
                for i, vismap in enumerate(vismap_list):
                    if "Diffuse material" in vismap:
                        new_name = "Trunk"
                    elif "Ground material" in vismap:
                        new_name = "Soil"
                    elif "Reference" in vismap:
                        new_name = vismap.split(" material")[0]
                    elif "Slab material" in vismap:
                        slab_mat_name = vismap.split("_0001")[0]
                        leaf_mat_map_idx = list_slab_name.index(slab_mat_name)
                        new_name = list_lotus_code[leaf_mat_map_idx]
                    else:
                        new_name = "ERROR in renaming abundance map"

                    map_name_indices[str(i)] = new_name

                    abundance_map = abundance_map_array[:, :, i]
                    print(
                        f"i:{i} = vismap:'{vismap}' renamed to '{new_name}', and ab_map shape is {abundance_map.shape}"
                    )

                    plt.imshow(abundance_map, cmap="viridis")
                    plt.title(f"Abundance {new_name}")
                    # plt.show()
                    image_name = f"Abundance {new_name}.png"
                    path = PH.join(path_dir_abundances, image_name)
                    logging.info(f"Saving abundance map visualization to '{path}'.")
                    plt.savefig(path, dpi=300)
                    plt.close()

                TH.write_dict_as_toml(
                    dictionary=map_name_indices,
                    directory=path_dir_abundances,
                    filename="map_name_indices.toml",
                )

                scene_number += 1


def run(runtime: RuntimeEnvironment):
    """Just a little run function to be called from main to keep it neat."""

    # logging.info("Dataset run started.")

    # generate_leaves()
    # solve_leaves(runtime=runtime, sims_to_solve_list=slab_sim_names)
    #
    # generate_random_prospect_leaves(slab_sim_name=slab_sim_name_pr, leaf_count=5)
    # solve_leaves(runtime=runtime, sims_to_solve_list=[slab_sim_name_pr])

    # rng = np.random.default_rng(1243567)
    # generate_forest_master(runtime=runtime, rng=rng)
    # lotus_to_hb(runtime)

    # Generate one ancestor scene that is used to spawn the rest
    # generate_ancestor(runtime)

    # Simulate lighting at Grenoble centrum coordinates at the last day of June at 13:00 local time
    # lighting.load_light(file_name="grenoble.txt")

    # For wet peat soil. Run this loop first and when the scenes are ok,
    # run the next one for dry sand
    for ss in scenes_and_signals:
        soil_type = "wet_peat_reflectance"
        generate_forest_variants(
            runtime=runtime,
            soil_name=soil_type,
            scene_and_signals=ss,
            generate_master=True,
            run_simulations=True,
            generate_resolutions=True,
        )

    # For dry sand soil
    for ss in scenes_and_signals:
        soil_type = "dry_sand_reflectance"
        generate_forest_variants(
            runtime=runtime,
            soil_name=soil_type,
            scene_and_signals=ss,
            generate_master=True,
            run_simulations=True,
            generate_resolutions=True,
        )

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


# Only for setting the ancestor scene once
# def generate_ancestor(runtime: RuntimeEnvironment):
#     sun_name = "grenoble_sun"
#     sky_name = "grenoble_sky"
#     soil_name = "wet_peat_reflectance"

# This is the master master that is used to spawn the highest resolution forests
# forest.init(
#     # leaves=leaves,
#     conf_type="m2m",
#     custom_forest_id=ancestor_scene,
#     soil_name=soil_name,
#     sun_file_name=sun_name,
#     sky_file_name=sky_name,
# )
#
# BC.generate_forest_control(
#     runtime=runtime, system_sim_name=ancestor_scene, global_master=False
# )

# BC.setup_system_sim_scene(
#     runtime=runtime, system_sim_name=ancestor_scene, leaf_name_list=[]
# )


def generate_forest_variants(
    runtime: RuntimeEnvironment,
    soil_name: str,
    scene_and_signals: dict,
    generate_master=False,
    generate_resolutions=False,
    run_simulations=False,
):

    sun_name = "grenoble_sun"
    sky_name = "grenoble_sky"

    theme = scene_and_signals["theme"]
    signal_tuples = scene_and_signals["signals"]
    leaves = []
    slab_mat_id = 1
    for signal_tuple in signal_tuples:
        set_dict = signal_tuple[0]
        signal_ids = signal_tuple[1]
        for signal_id in signal_ids:
            leaves.append(
                (set_dict["slab_sim_name"], signal_id, f"Slab material {slab_mat_id}")
            )
            slab_mat_id += 1

    use_theme = theme
    if soil_name == "dry_sand_reflectance":
        dry_theme = theme.replace("WP", "DS")
        use_theme = dry_theme

    # This is the master master that is used to spawn the highest resolution forests
    # if generate_master:
    #     if soil_name == "wet_peat_reflectance":
    #         forest.init(
    #             leaves=leaves,
    #             conf_type="m2m",
    #             copy_forest_id=ancestor_scene,  # This is a copy of the ancestor
    #             custom_forest_id=theme,  # Copied to the theme name
    #             soil_name=soil_name,
    #             sun_file_name=sun_name,
    #             sky_file_name=sky_name,
    #         )
    #     elif soil_name == "dry_sand_reflectance":
    #         # Instead of generating the dry sand version from scratch, we copy the wet peat version
    #         forest.init(
    #             leaves=leaves,
    #             conf_type="m2m",
    #             copy_forest_id=theme,  # The other soil is copied from the first soil--not from the ancestor
    #             custom_forest_id=use_theme,  # And named accordingly
    #             soil_name=soil_name,
    #             sun_file_name=sun_name,
    #             sky_file_name=sky_name,
    #         )
    #
    #         material_dict = copy_lotus_data(signal_tuples, dst_sys_sim_name=use_theme)
    #         TH.write_dict_as_toml(
    #             material_dict,
    #             PH.directory_system_simulation(use_theme),
    #             filename="leaf_material_map",
    #         )
    #
    #         leaf_name_list = material_dict["slab_material_names"]
    #         BC.setup_system_sim_scene(
    #             runtime=runtime,
    #             system_sim_name=use_theme,
    #             leaf_name_list=leaf_name_list,
    #         )
    #     else:
    #         raise ValueError(f"Unknown soil type {soil_name}.")

    # This has to be generated when copying from master
    BC.generate_forest_control(
        runtime=runtime, system_sim_name=use_theme, global_master=False
    )

    if generate_resolutions:
        high_level_name = run_next_resolution(
            runtime=runtime,
            resolution=1024,
            high_level_name=use_theme,
            leaves=leaves,
            soil_name=soil_name,
            sun_name=sun_name,
            sky_name=sky_name,
            do_copy=True,
            run_setup_and_render=run_simulations,
            signal_tuples=signal_tuples,
        )
        run_next_resolution(
            runtime=runtime,
            resolution=256,
            high_level_name=high_level_name,
            leaves=leaves,
            soil_name=soil_name,
            sun_name=sun_name,
            sky_name=sky_name,
            do_copy=True,
            run_setup_and_render=run_simulations,
            signal_tuples=signal_tuples,
        )
        run_next_resolution(
            runtime=runtime,
            resolution=64,
            high_level_name=high_level_name,
            leaves=leaves,
            soil_name=soil_name,
            sun_name=sun_name,
            sky_name=sky_name,
            do_copy=True,
            run_setup_and_render=run_simulations,
            signal_tuples=signal_tuples,
        )
        run_next_resolution(
            runtime=runtime,
            resolution=16,
            high_level_name=high_level_name,
            leaves=leaves,
            soil_name=soil_name,
            sun_name=sun_name,
            sky_name=sky_name,
            do_copy=True,
            run_setup_and_render=run_simulations,
            signal_tuples=signal_tuples,
        )
        run_next_resolution(
            runtime=runtime,
            resolution=4,
            high_level_name=high_level_name,
            leaves=leaves,
            soil_name=soil_name,
            sun_name=sun_name,
            sky_name=sky_name,
            do_copy=True,
            run_setup_and_render=run_simulations,
            signal_tuples=signal_tuples,
        )


def run_next_resolution(
    runtime,
    resolution: int,
    high_level_name: str,
    leaves,
    soil_name: str,
    sun_name: str,
    sky_name: str,
    signal_tuples,
    do_copy=False,
    run_setup_and_render: bool = False,
):

    theme = high_level_name.split(sep="_")[0]
    current_level_name = f"{theme}_{resolution}"

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

        material_dict = copy_lotus_data(
            signal_tuples, dst_sys_sim_name=current_level_name
        )
        leaf_name_list = material_dict["slab_material_names"]
        TH.write_dict_as_toml(
            material_dict,
            PH.directory_system_simulation(current_level_name),
            filename="leaf_material_map",
        )

        sys_sim_path = PH.directory_system_simulation(current_level_name)
        scene_control = TH.read_toml_as_dict(
            directory=sys_sim_path, filename="system_sim_control"
        )
        scene_control["Images"]["hsi_resolution_x"] = resolution
        scene_control["Images"]["hsi_resolution_y"] = resolution
        scene_control["Images"]["rgb_resolution_x"] = resolution
        scene_control["Images"]["rgb_resolution_y"] = resolution

        scene_control["Images"]["walker_resolution_x"] = 1024
        scene_control["Images"]["walker_resolution_y"] = 512
        scene_control["Images"]["sleeper_resolution_x"] = 1024
        scene_control["Images"]["sleeper_resolution_y"] = 512
        scene_control["Images"]["tree_preview_resolution_x"] = 1024
        scene_control["Images"]["tree_preview_resolution_y"] = 512

        sample_count = 32
        if resolution == 256:
            sample_count = 128
        elif resolution == 64:
            sample_count = 512
        elif resolution == 16:
            sample_count = 2048
        elif resolution == 4:
            sample_count = 8192

        scene_control["Rendering"]["sample_count_hsi"] = sample_count
        scene_control["Rendering"]["sample_count_rbg"] = sample_count

        TH.write_dict_as_toml(
            directory=sys_sim_path,
            dictionary=scene_control,
            filename="system_sim_control",
        )

    if run_setup_and_render:
        BC.setup_system_sim_scene(
            runtime=runtime,
            system_sim_name=current_level_name,
            leaf_name_list=leaf_name_list,
        )

        # Visibility maps and previews only for the high res cube.
        if resolution >= 1024:

            BC.render_forest(
                runtime=runtime,
                system_sim_name=current_level_name,
                render_mode="preview",
            )

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

        if resolution >= 1024:
            # For high res, the white is inferred from the data.
            CH.construct_envi_cube(system_sim_name=current_level_name)
        else:
            # For lower res, use the white reference from the high res data.
            CH.construct_envi_cube(
                system_sim_name=current_level_name,
                system_sim_name_for_white_signal=high_level_name,
            )

    return current_level_name


def copy_lotus_data(signal_tuples, dst_sys_sim_name: str):
    # Copy original LOTUS leaf data and slab simulation results so that the system
    #   simulation directory is self-contained

    # Slab materials in a list of tuples (slab_material_name, lotus_code)
    slab_material_names = []
    lotus_codes = []

    slab_material_index = 1
    for signal_tuple in signal_tuples:
        set_dict = signal_tuple[0]
        slab_sim_name = set_dict["slab_sim_name"]
        signal_ids = signal_tuple[1]
        for signal_id in signal_ids:
            slab_sim_dir = PH.directory_slab_simulation(slab_sim_name=slab_sim_name)
            signal_res_plot_src = PH.file_signal_result_plot(
                slab_sim_name=slab_sim_name, signal_id=signal_id
            )
            signal_res_toml_src = PH.file_signal_result(
                slab_sim_name=slab_sim_name, signal_id=signal_id
            )
            lotus_filename_stump = f"{set_dict['lotus_codes'][signal_id]}"

            slab_material_names.append(f"Slab material {slab_material_index}")
            lotus_codes.append(f"{lotus_filename_stump}")
            slab_material_index += 1

            lotus_toml_filename = f"{lotus_filename_stump}.toml"
            lotus_jpg_filename = f"{lotus_filename_stump}.JPG"
            lotus_analysis_toml_src = PH.join(slab_sim_dir, lotus_toml_filename)
            lotus_analysis_jpg_src = PH.join(slab_sim_dir, lotus_jpg_filename)
            if (
                not os.path.exists(signal_res_plot_src)
                or not os.path.exists(signal_res_toml_src)
                or not os.path.exists(lotus_analysis_toml_src)
                or not os.path.exists(lotus_analysis_jpg_src)
            ):
                raise FileNotFoundError(
                    f"Missing some data files of slab simulation {slab_sim_name} signal ID {signal_id}."
                )
            new_signal_result_toml_name = (
                f"{lotus_filename_stump}_{signal_id}_slab_sim_result.toml"
            )
            new_signal_result_plot_name = (
                f"{lotus_filename_stump}_{signal_id}_slab_sim_result.png"
            )
            shutil.copy(
                signal_res_toml_src,
                PH.join(
                    PH.directory_system_simulation(dst_sys_sim_name),
                    new_signal_result_toml_name,
                ),
            )
            shutil.copy(
                signal_res_plot_src,
                PH.join(
                    PH.directory_system_simulation(dst_sys_sim_name),
                    new_signal_result_plot_name,
                ),
            )
            shutil.copy(
                lotus_analysis_toml_src,
                PH.join(
                    PH.directory_system_simulation(dst_sys_sim_name),
                    lotus_toml_filename,
                ),
            )
            shutil.copy(
                lotus_analysis_jpg_src,
                PH.join(
                    PH.directory_system_simulation(dst_sys_sim_name),
                    lotus_jpg_filename,
                ),
            )

    material_dict = {
        "slab_material_names": slab_material_names,
        "lotus_codes": lotus_codes,
    }
    return material_dict


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

    # BC.render_forest(
    #     runtime=runtime,
    #     system_sim_name=system_sim_name_slave,
    #     render_mode="preview",
    #     silent=False,
    # )
    #
    # BC.render_forest(
    #     runtime=runtime,
    #     system_sim_name=system_sim_name_slave,
    #     render_mode="spectral",
    #     silent=False,
    # )
    #
    # BC.render_forest(
    #     runtime=runtime,
    #     system_sim_name=system_sim_name_slave,
    #     render_mode="visibility",
    #     silent=False,
    # )
    #
    # CH.construct_envi_cube(system_sim_name=system_sim_name_slave)

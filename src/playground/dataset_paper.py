"""

This module is used to generate leaves for the dataset paper

"""

import logging
import os.path

# import pandas as pd
import spectral
from scipy.io import savemat
import yaml

import numpy as np
import shutil
from matplotlib import pyplot as plt

from src.setup.runtime_environment import RuntimeEnvironment
from src.slab_model import interface as SI
from src.system_simulation import forest, lighting
from src.rendering import blender_control as BC
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


# Noisy bands
remove = np.concatenate((np.arange(192, 202), np.arange(284, 307)))


def run(runtime: RuntimeEnvironment):
    """Just a little run function to be called from main to keep the main neat."""

    # logging.info("Dataset run started.")

    # Solve LOTUS leaves material parameters to be used in canopy sim.
    # lotus_to_hb(runtime)

    # Generate one ancestor scene that is used to spawn the rest
    # generate_ancestor(runtime)

    # Generate and run all forest simulations.
    # run_forest_simulations(runtime)
    # generate_and_run_forest_simulations(
    #     runtime=runtime, do_copy=True, run_setup=True, run_render=True
    # )

    # calculate_endmembers_and_abundances()
    hysuppify_all()


def generate_and_run_forest_simulations(
    runtime: RuntimeEnvironment,
    do_copy=False,
    run_render=False,
    run_setup=False,
):
    """Top level function to generate and run all forest simulations.

    :param runtime: Runtime environment
    :param do_copy: If True, the scene is copied from the theme level scene named like FWP1, FDS2, etc.
    :param run_render: If True, renders the scenes after generation. This takes a lot of time.
    :param run_setup: If True, runs just the setup unless `run_render` is also True
    :return: None
    """

    # For wet peat soil. Run this loop first and when the scenes are ok,
    # run the next one for dry sand
    for ss in scenes_and_signals:
        soil_type1 = "wet_peat_reflectance"
        soil_type2 = "dry_sand_reflectance"
        generate_and_run_forest_variants(
            runtime=runtime,
            soil_name=soil_type1,
            scene_and_signals=ss,
            do_copy=do_copy,
            run_render=run_render,
            run_setup=run_setup,
        )

        generate_and_run_forest_variants(
            runtime=runtime,
            soil_name=soil_type2,
            scene_and_signals=ss,
            do_copy=do_copy,
            run_render=run_render,
            run_setup=run_setup,
        )


def generate_and_run_forest_variants(
    runtime: RuntimeEnvironment,
    soil_name: str,
    scene_and_signals: dict,
    do_copy=False,
    run_render=False,
    run_setup=False,
):
    """Generates and runs forest simulations in all resolutions for given soil type and scene.

    :param runtime: Runtime environment
    :param soil_name: Soil name
    :param scene_and_signals: Single item from scenes_and_signals dict.
    :param do_copy: If True, the scene is copied from the scene according to the theme in `scene_and_signals`.
        Only forest initialization and material copying is done here.
    :param run_render: If True, runs both the setup and rendering after scene generation
    :param run_setup: If True, runs just the setup unless `run_render` is also True
    :return: None
    """

    sun_name = "grenoble_sun"
    sky_name = "grenoble_sky"

    # The theme in the scene_and_signals is always for wet peat such as "FWP1"
    theme = scene_and_signals["theme"]

    # The theme_with_soil is the actual theme to be used depending on soil type
    theme_with_soil = theme
    if soil_name == "dry_sand_reflectance":
        dry_theme = theme.replace("WP", "DS")
        theme_with_soil = dry_theme

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

    # First do the full res separately and save its name
    high_level_name = generate_next_resolution_scene(
        runtime=runtime,
        resolution=1024,
        high_level_name=theme_with_soil,
        leaves=leaves,
        soil_name=soil_name,
        sun_name=sun_name,
        sky_name=sky_name,
        signal_tuples=signal_tuples,
        do_copy=do_copy,
        run_setup=run_setup,
        run_render=run_render,
    )

    # Then loop the low resolution scenes
    for res in [256, 64, 16, 4]:
        generate_next_resolution_scene(
            runtime=runtime,
            resolution=res,
            high_level_name=high_level_name,
            leaves=leaves,
            soil_name=soil_name,
            sun_name=sun_name,
            sky_name=sky_name,
            signal_tuples=signal_tuples,
            do_copy=do_copy,
            run_setup=run_setup,
            run_render=run_render,
        )

    #####
    ##### This snippet should not be used. It is left here for reference only. It generates the
    ##### theme level scenes from the one single ancestor which doesn't contain any of the geometry
    ##### that must be done by hand.
    #####
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
    #             custom_forest_id=theme_with_soil,  # And named accordingly
    #             soil_name=soil_name,
    #             sun_file_name=sun_name,
    #             sky_file_name=sky_name,
    #         )
    #
    #         material_dict = copy_lotus_data(signal_tuples, dst_sys_sim_name=theme_with_soil)
    #         TH.write_dict_as_toml(
    #             material_dict,
    #             PH.directory_system_simulation(theme_with_soil),
    #             filename="leaf_material_map",
    #         )
    #
    #         leaf_name_list = material_dict["slab_material_names"]
    #         BC.setup_system_sim_scene(
    #             runtime=runtime,
    #             system_sim_name=theme_with_soil,
    #             leaf_name_list=leaf_name_list,
    #         )
    #     else:
    #         raise ValueError(f"Unknown soil type {soil_name}.")
    # # This has to be generated when copying from master
    # BC.generate_forest_control(
    #     runtime=runtime, system_sim_name=theme_with_soil, global_master=False
    # )


def generate_next_resolution_scene(
    runtime,
    resolution: int,
    high_level_name: str,
    leaves,
    soil_name: str,
    sun_name: str,
    sky_name: str,
    signal_tuples,
    do_copy=False,
    run_setup=False,
    run_render=False,
):
    """Generates the next resolution level of the forest simulation.

    :param runtime: Runtime environment
    :param resolution: Resolution to generate one of (1024, 256, 64, 16, 4)
    :param high_level_name: Name of the high level scene to copy from. For full res scene, this is the theme name.
    :param leaves: List of leaves to use in the forest initialization. Constant within a theme.
    :param soil_name: The name of the soil file
    :param sun_name: The name of the sun file
    :param sky_name: The name of the sky file
    :param signal_tuples: List of signal tuples used to copy the LOTUS data.
    :param do_copy: If True, the scene is copied from the scene indicated by `high_level_name`
        and modified according to given resolution. Only forest initialization and material copying is done here.
    :param run_setup: If True, the scene setup is run. Can be used separately from rendering.
    :param run_render: If True, both the scene setup and rendered is run. Also constructs the ENVI cube after rendering.
    :return: The name of the current level that was just generated.
    """

    theme = high_level_name.split(sep="_")[0]
    current_level_name = f"{theme}_{resolution}"
    leaf_material_map_name = "leaf_material_map"

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
        TH.write_dict_as_toml(
            material_dict,
            PH.directory_system_simulation(current_level_name),
            filename=leaf_material_map_name,
        )

        sys_sim_path = PH.directory_system_simulation(current_level_name)
        scene_control = TH.read_toml_as_dict(
            directory=sys_sim_path, filename="system_sim_control"
        )
        scene_control["Images"]["hsi_resolution_x"] = resolution
        scene_control["Images"]["hsi_resolution_y"] = resolution
        scene_control["Images"]["rgb_resolution_x"] = resolution
        scene_control["Images"]["rgb_resolution_y"] = resolution

        # These preview resolutions are kept constant (they are only rendered for high res scene anyways)
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

    if run_setup or run_render:
        material_dict = TH.read_toml_as_dict(
            PH.directory_system_simulation(current_level_name),
            filename=leaf_material_map_name,
        )
        leaf_name_list = material_dict["slab_material_names"]
        BC.setup_system_sim_scene(
            runtime=runtime,
            system_sim_name=current_level_name,
            leaf_name_list=leaf_name_list,
        )

    if run_render:
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

        # Spectral bands are rendered for all resolutions
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


def calculate_endmembers_and_abundances(clear_old=True):
    """

    1. Calculates endmembers from full resolution cube.
    1.1 This may reduce the cube band-wise as noisy bands are removed and
        also the number of endmembers is lower than the number of visibility maps
        as we ignore the reference materials and materials that were never spawned to
        the scene. We will not modify the original raw reflectance cube but only save
        a new one to be run in HySUPP.
    2. Calculate abundances for the full resolution cube. The number of abundances is
        based on the endmembers calculated in step 1.
    3. Repeat for low resolution cubes but use the endmembers from full resolution cube.

    :param clear_old: If true, old abundance files are deleted before
        calculating new ones.
    :return:
    """

    for ss in scenes_and_signals:

        theme_wp = ss["theme"]  # soil: wet peat
        theme_ds = theme_wp.replace("WP", "DS")  # soil: dry sand

        for theme in [theme_wp, theme_ds]:

            sys_sim_name_full_res = f"{theme}_1024"

            # This will be filled by the first iteration and stays the same for other resolutions
            accepted_visibility_names = None

            # Assigned to be the full res scene
            path_visibility = None

            for res in [1024, 256, 64, 16, 4]:
                sys_sim_name = f"{theme}_{res}"
                dir_sys_sim = PH.directory_system_simulation(sys_sim_name)

                if not os.path.exists(dir_sys_sim):
                    raise FileNotFoundError(
                        f"System simulation {sys_sim_name} does not exist."
                    )

                path_to_img = PH.file_spectral_cube(
                    system_sim_name=sys_sim_name, file_type="data"
                )
                path_to_hdr = PH.file_spectral_cube(
                    system_sim_name=sys_sim_name, file_type="header"
                )

                # Open cube. This is needed for ground_truth_endmembers function
                cube = spectral.envi.open(path_to_hdr, path_to_img)

                # Load cube to memory
                Y_cube = cube.load()
                # Delete noisy bands
                Y_cube = np.delete(Y_cube, remove, axis=2)

                # Only calculate endmembers for the full resolution cube
                if res == 1024:

                    # Load visibility maps
                    path_visibility, visibility_names = get_vismaps(sys_sim_name)

                    # Get ground truth endmembers with empty visibility maps ignored
                    material_means_array, accepted_visibility_names = (
                        CD.ground_truth_endmembers(
                            Y_cube,
                            visibility_maps_dir=path_visibility,
                            visibility_names=visibility_names,
                        )
                    )

                    # Create and save a dict to be saved in toml file for endmember-index mapping
                    endmember_names = {}

                    endmember_idx = 0
                    for i, visibility_name in enumerate(accepted_visibility_names):
                        endmember_names[f"{endmember_idx}"] = visibility_name
                        endmember_idx += 1

                    # Create endmember directory if it does not exist
                    path_dir_em = path_dir_endmembers(sys_sim_name=sys_sim_name)
                    if not os.path.exists(path_dir_em):
                        os.makedirs(path_dir_em)

                    if clear_old:
                        for file in os.listdir(path_dir_em):
                            file_path = PH.join(path_dir_em, file)
                            os.remove(file_path)

                    # Save endmember array to disk
                    np.save(path_file_endmembers(sys_sim_name), material_means_array)

                    # And write the names mapping file
                    TH.write_dict_as_toml(
                        dictionary=endmember_names,
                        directory=path_dir_endmembers(sys_sim_name),
                        filename="endmember_names.toml",
                    )

                    plot_endmembers(
                        sys_sim_name=sys_sim_name, save_thumbnail=True, dont_show=True
                    )

                    #### This is the end of endmembers that is only for the full res cube. Continue with abundances ####

                # Scaling factors for different resolutions. Set to 1 for full res.
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
                    visibility_names=accepted_visibility_names,
                )

                # Create directory for abundance maps if it does not exist
                path_dir_abundance = path_dir_abundances(sys_sim_name=sys_sim_name)
                if not os.path.exists(path_dir_abundance):
                    os.makedirs(path_dir_abundance)

                # Clear old files. Useful when recalculating abundances.
                if clear_old:
                    for file in os.listdir(path_dir_abundance):
                        file_path = PH.join(path_dir_abundance, file)
                        os.remove(file_path)

                abundance_map_save_path = PH.join(path_dir_abundance, f"abundances.npy")

                # Save the full abundance map array as a single numpy file
                np.save(abundance_map_save_path, abundance_map_array)

                map_name_indices = {
                    "note": "Mapping of abundance map indices to human readable names. "
                    "When you load the abundance map numpy array, you can find "
                    "a specific map by its index in this file and use the associated name."
                }

                # Load leaf material name mapping to rename abundance maps
                leaf_mat_name_map = TH.read_toml_as_dict(
                    directory=PH.directory_system_simulation(sys_sim_name_full_res),
                    filename="leaf_material_map.toml",
                )
                list_slab_name = leaf_mat_name_map["slab_material_names"]
                list_lotus_code = leaf_mat_name_map["lotus_codes"]

                # Rename material names to more human readable names and save visualizations
                for i, vismap in enumerate(accepted_visibility_names):
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
                    logging.debug(
                        f"i:{i} = vismap:'{vismap}' renamed to '{new_name}', and ab_map shape is {abundance_map.shape}"
                    )

                    plt.close("all")
                    plt.imshow(abundance_map, cmap="viridis")
                    plt.title(f"Abundance {new_name}")
                    # plt.show()
                    image_name = f"Abundance {new_name}.png"
                    path = PH.join(path_dir_abundance, image_name)
                    logging.info(f"Saving abundance map visualization to '{path}'.")
                    plt.savefig(path, dpi=300)
                    plt.close()

                TH.write_dict_as_toml(
                    dictionary=map_name_indices,
                    directory=path_dir_abundance,
                    filename="map_name_indices.toml",
                )


def hysuppify_all():
    """Reformulates all scenes to a format that can be passed directly to HySUPP."""

    for ss in scenes_and_signals:
        theme_wp = ss["theme"]  # soil: wet peat
        theme_ds = theme_wp.replace("WP", "DS")  # soil: dry sand

        for theme in [theme_wp, theme_ds]:

            sys_sim_name_full_res = f"{theme}_1024"
            dir_sys_sim_full_res = PH.directory_system_simulation(sys_sim_name_full_res)

            if not os.path.exists(dir_sys_sim_full_res):
                logging.warning(
                    f"Full resolution system simulation {dir_sys_sim_full_res} does not exist. Skipping low resolution versions too."
                )
                continue

            for res in [1024, 256, 64, 16, 4]:
                sys_sim_name = f"{theme}_{res}"
                dir_sys_sim = PH.directory_system_simulation(sys_sim_name)

                if not os.path.exists(dir_sys_sim):
                    logging.warning(
                        f"System simulation {sys_sim_name} does not exist. Skipping this resolution."
                    )
                    continue

                E = load_endmembers(sys_sim_name_full_res)  # Always load from full res
                A = load_abundances(sys_sim_name)
                save_hysupp_cube(sys_sim_name, E, A)


def save_hysupp_cube(sys_sim_name, E, A):
    """Saves the data in HySUPP format."""

    # Transform the data to right dimensions
    # Checked the right dimensions from DC1.mat data.
    H, W, p = A.shape
    A = A.reshape(H * W, p).T

    path_to_img = PH.file_spectral_cube(system_sim_name=sys_sim_name, file_type="data")
    path_to_hdr = PH.file_spectral_cube(
        system_sim_name=sys_sim_name, file_type="header"
    )

    # Open cube. This is needed for ground_truth_endmembers function
    cube = spectral.envi.open(path_to_hdr, path_to_img)

    # Load cube to memory
    Y_cube = cube.load()
    # Delete noisy bands
    Y_cube = np.delete(Y_cube, remove, axis=2)

    H, W, L = Y_cube.shape
    N = H * W
    Y = Y_cube.reshape(H * W, L).T
    logging.info(f"Y shape: {Y.shape}, E shape: {E.shape}, A shape: {A.shape}")

    dataset = {
        "Y": Y,
        "H": H,
        "W": W,
        "L": L,
        "N": N,
        "E": E,  # Ground truth endmembers
        "A": A,  # Ground truth abundances
        "p": p,  # Number of endmembers
    }

    # data_class_name = "src.data.base.RealHSI"
    data_class_name = "src.data.base.HSIWithGT"
    dataset_name = sys_sim_name

    path_dir_hysuppified = PH.join(PH.directory_project_root(), "Hysuppified")
    path_dir_hysupp_data = PH.join(path_dir_hysuppified, "data")
    path_dir_hysupp_config = PH.join(path_dir_hysuppified, "config", "data")
    if not os.path.exists(path_dir_hysuppified):
        os.makedirs(path_dir_hysuppified)
    if not os.path.exists(path_dir_hysupp_data):
        os.makedirs(path_dir_hysupp_data)
    if not os.path.exists(path_dir_hysupp_config):
        os.makedirs(path_dir_hysupp_config)

    # Create and save mat and yaml files to right subdirectories.
    # save_path_mat = "./HySUPP/data/" + dataset_name + ".mat"
    save_path_mat = PH.join(path_dir_hysupp_data, f"{dataset_name}.mat")
    savemat(save_path_mat, dataset)
    logging.info(f"Saved mat to {save_path_mat}")

    # save_path_yaml = "./HySUPP/config/data/" + dataset_name + ".yaml"
    save_path_yaml = PH.join(path_dir_hysupp_config, f"{dataset_name}.yaml")
    dataset_yaml = {
        "name": data_class_name,
        "dataset": dataset_name,
        "p": p,
        "data_dir": "${DATA_dir}",
        "figs_dir": "${FIGS_dir}",
    }

    with open(f"{save_path_yaml}", "w") as file:
        yaml.dump(dataset_yaml, file, sort_keys=False)

    logging.info(f"Saved yaml to {save_path_yaml}")


def plot_endmembers(sys_sim_name, save_thumbnail=False, dont_show=False):
    E = load_endmembers(sys_sim_name)
    endmember_names = TH.read_toml_as_dict(
        directory=path_dir_endmembers(sys_sim_name), filename="endmember_names.toml"
    )
    leaf_material_map = TH.read_toml_as_dict(
        PH.directory_system_simulation(sys_sim_name), filename="leaf_material_map.toml"
    )
    slab_material_names = leaf_material_map["slab_material_names"]
    lotus_codes = leaf_material_map["lotus_codes"]

    plt.close("all")

    # Decide the line style based on the material name
    for index, name in endmember_names.items():
        endmember = E[:, int(index)]
        plot_label = name
        if "Reference" in name:
            line_style = "dashed"
            splitted = name.split(" ")
            plot_label = splitted[0] + " " + splitted[1]
        elif "Diffuse" in name:
            line_style = "dotted"
            plot_label = "Trunk"
        elif "Ground" in name:
            line_style = "dashdot"
            plot_label = "Soil"
        else:
            for x, slab_material_name in enumerate(slab_material_names):
                if slab_material_name in name:
                    plot_label = lotus_codes[x]
            line_style = "solid"

        plt.rcParams["figure.figsize"] = (15, 9)
        plt.plot(endmember, label=plot_label, ls=line_style)
        plt.legend()

    if save_thumbnail:
        save_resolution = 100
        p = path_dir_endmembers(sys_sim_name)
        image_name = f"endmembers.png"
        save_path = PH.join(p, image_name)
        logging.info(f"Saving endmember plot to '{save_path}'.")
        plt.savefig(save_path, dpi=save_resolution)

    if not dont_show:
        plt.show()


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


def get_vismaps(sys_sim_name):
    """Returns visibility maps for given simulation.

    :returns: tuple (path, list), where path is a path to the visibility
        maps directory, and the list is a list of strings that contain the
        file names of the visibility maps in that directory.
    """

    path_visibility = PH.directory_system_rend_visibility_maps(
        system_sim_name=sys_sim_name
    )

    if not os.path.exists(path_visibility):
        raise FileNotFoundError(
            f"Visibility map directory {path_visibility} does not exist."
        )
    vismap_list = PH.list_visibility_maps(system_sim_name=sys_sim_name)
    # Exclude possible other files and rgb previews
    vismap_list = [
        file_name
        for file_name in vismap_list
        if (file_name.endswith(".tif") and not "rgb_preview" in file_name)
    ]
    return path_visibility, vismap_list


def recalculate_cubes():
    """Calculates reflectance cubes from the raw rendered bands.

    Can be used if the reflectance calculation is changed but no other modifications
    are needed. Remember to recalculate endmembers and abundances after this.
    """

    for ss in scenes_and_signals:
        theme_wp = ss["theme"]  # soil: wet peat
        theme_ds = theme_wp.replace("WP", "DS")  # soil: dry sand

        for theme in [theme_wp, theme_ds]:

            sys_sim_name_full_res = f"{theme}_1024"
            CH.construct_envi_cube(system_sim_name=sys_sim_name_full_res)

            for res in [256, 64, 16, 4]:
                sys_sim_name = f"{theme}_{res}"
                CH.construct_envi_cube(
                    system_sim_name=sys_sim_name,
                    system_sim_name_for_white_signal=sys_sim_name_full_res,
                )


def load_endmembers(sys_sim_name):
    """Load endmembers as a numpy array."""
    p = path_file_endmembers(sys_sim_name)
    if not os.path.exists(p):
        raise RuntimeError(f"Endmember file '{p}' does not exist.")
    E = np.load(p)
    return E


def load_abundances(sys_sim_name):
    p = path_file_abundances(sys_sim_name)
    if not os.path.exists(p):
        raise RuntimeError(f"Abundances file '{p}' does not exist.")
    A = np.load(p)
    return A


def path_dir_endmembers(sys_sim_name):
    """Returns directory where the endmembers are saved."""
    path_dir_endmembu = PH.join(
        PH.directory_system_simulation(system_sim_name=sys_sim_name), "Endmembers"
    )
    return path_dir_endmembu


def path_dir_abundances(sys_sim_name):
    """Returns directory where the endmembers are saved."""
    path_dir_endmembu = PH.join(
        PH.directory_system_simulation(system_sim_name=sys_sim_name), "Abundance maps"
    )
    return path_dir_endmembu


def path_file_endmembers(sys_sim_name):
    path_dir = path_dir_endmembers(sys_sim_name)
    path_file = PH.join(path_dir, "endmembers.npy")
    return path_file


def path_file_abundances(sys_sim_name):
    path_dir = path_dir_abundances(sys_sim_name)
    path_file = PH.join(path_dir, "abundances.npy")
    return path_file


#
# DO NOT USE BUT DO NOT DELETE EITHER
#
# def generate_ancestor(runtime: RuntimeEnvironment):
#     """Generates the ancestor scene that is used to spawn the rest of the dataset.
#
#     Should not be used anymore after the theme scenes have been created by hand.
#     """
#     # Only for setting the ancestor scene once
#     sun_name = "grenoble_sun"
#     sky_name = "grenoble_sky"
#     soil_name = "wet_peat_reflectance"
#
#     # This is the master master that is used to spawn the highest resolution forests
#     forest.init(
#         # leaves=leaves,
#         conf_type="m2m",
#         custom_forest_id=ancestor_scene,
#         soil_name=soil_name,
#         sun_file_name=sun_name,
#         sky_file_name=sky_name,
#     )
#
#     BC.generate_forest_control(
#         runtime=runtime, system_sim_name=ancestor_scene, global_master=False
#     )
#
#     BC.setup_system_sim_scene(
#         runtime=runtime, system_sim_name=ancestor_scene, leaf_name_list=[]
#     )


def run_late_resampling(runtime: RuntimeEnvironment):
    # In case you forgot to resample them earlier, they have to be solved again.
    # Just leaving this snippet for future reference.
    for lotus_sample_dict in lotus_sample_dicts:
        slab_sim_name = lotus_sample_dict["slab_sim_name"]
        SI.resample_slab_sim_target(
            slab_sim_name=slab_sim_name, range_start=400, range_end=2500, resolution=5
        )  # resample leaf spectra

        SI.solve_leaf_material_parameters(
            runtime=runtime,
            slab_sim_name=slab_sim_name,
            clear_old_results=True,
            range_start=400,
            range_end=2500,
            resolution=5,
            solver_dirname="Iterative slab",
        )  # run slab simulation


def copy_lotus_data(signal_tuples, dst_sys_sim_name: str):
    """Copies the LOTUS leaf data and slab simulation results to create self-contained system simulation directory.

    :param signal_tuples:
    :param dst_sys_sim_name:
    :return:
    """

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

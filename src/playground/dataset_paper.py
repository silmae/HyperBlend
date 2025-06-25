"""

This module is used to generate leaves for the dataset paper

"""

import logging
import numpy as np

from setup.runtime_environment import RuntimeEnvironment
from src.slab_model import interface as SI
from src.system_simulation import forest
from rendering import blender_control as BC
from src.data import cube_handling as CH


slab_sim_names = ["Manitoba Maple", "American Elm", "Crab apple"]
slab_sim_name_pr = "dataset_paper_prospect_leaves"


def run(runtime: RuntimeEnvironment):
    """Just a little run function to be called from main to keep it neat."""

    # logging.info("Dataset run started.")

    # generate_leaves()
    # solve_leaves(runtime=runtime, slab_sim_names=slab_sim_names)

    # generate_random_prospect_leaves(slab_sim_name=slab_sim_name_pr, leaf_count=5)
    # solve_leaves(runtime=runtime, slab_sim_names=[slab_sim_name_pr])

    rng = np.random.default_rng(1243567)
    generate_forest_master(runtime=runtime, rng=rng)


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

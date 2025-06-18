"""

This module is used to generate leaves for the dataset paper

"""

import logging

from setup.runtime_environment import RuntimeEnvironment
from src.slab_model import interface as SI
from src.system_simulation import forest

slab_sim_names = ["Crab apple", "Manitoba Maple", "American Elm"]


def run(runtime: RuntimeEnvironment):
    """Just a little run function to be called from main to keep it neat."""

    logging.info("Dataset run started.")
    generate_leaves()
    solve_leaves(runtime=runtime)


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


def solve_leaves(runtime: RuntimeEnvironment):
    """Solve leaf parameters."""

    for slab_sim_name in slab_sim_names:
        SI.solve_leaf_material_parameters(
            runtime=runtime,
            slab_sim_name=slab_sim_name,
            resolution=5,
            range_start=400,
            range_end=900,
            solver="nn",
        )

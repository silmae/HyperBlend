"""
TODO this whole thing should be reworked to solve reflective diffuse materials.
    If not already implemented elswhere. Otherwise, this can be destroyed or
    moved under drafting.

"""

import data.path_handling
from src import plotter, constants as C
from src.rendering import blender_control as BC
from src.slab_simulation import slab_commons as LC
from src.utils import data_utils as DU
from src.data import file_handling as FH, path_handling as PH
from src.setup.runtime_environment import RuntimeEnvironment


def run(runtime: RuntimeEnvironment, data_exits=False):
    """Run virtual reflectance lab. The result will be plotted to project's root directory.

    This is only to show that if the sun power is more than 4 W/m2, a completely white
    diffuse surface will burn to white.

    :param data_exits: If True, no new data is created. This is merely for replotting. Default is False.
    """

    set_name = "reflectance_lab"
    LC.initialize_directories(slab_sim_name=set_name, clear_old_results=True)

    powers = [4, 5, 6]
    reflectance = []
    HSV_values = list(range(101))
    for sun_power in powers:
        sample_dir = PH.directory_result_signal(
            slab_sim_name=set_name, signal_id=sun_power
        )
        if not data_exits:
            BC.run_reflectance_lab(
                rend_base_path=sample_dir,
                dry_run=False,
                sun_power=sun_power,
                runtime=runtime,
            )
        vals = []
        for value in HSV_values:
            p = PH.find_slab_opt_render_by_wl(
                wl=value,
                mode=C.target_type_slab,
                imaging_type=C.imaging_type_refl,
                base_path=sample_dir,
            )
            vals.append(DU.get_rend_as_mean(p))

        reflectance.append(vals)

    plotter.plot_reflectance_lab(
        HSV_value=HSV_values,
        reflectance=reflectance,
        powers=powers,
        plot_name="diffuce_reflectance",
    )

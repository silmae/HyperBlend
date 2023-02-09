
from src import constants as C, plotter
from src.rendering import blender_control as BC
from src.leaf_model import leaf_commons as LC
from src.utils import data_utils as DU
from src.data import file_handling as FH, path_handling as PH


def run(data_exits=False):
    """Run virtual reflectance lab. The result will be plotted to project's root directory.

    :param data_exits:
        If True, no new data is created. This is merely for replotting. Default is False.
    """

    set_name = 'reflectance_lab'
    LC.initialize_directories(set_name=set_name, clear_old_results=True)

    powers = [4,5,6]
    reflectance = []
    HSV_values = list(range(101))
    for sun_power in powers:
        sample_dir = PH.path_directory_sample(set_name=set_name, sample_id=sun_power)
        if not data_exits:
            BC.run_reflectance_lab(rend_base_path=sample_dir, dry_run=False, sun_power=sun_power)
        vals = []
        for value in HSV_values:
            p = FH.search_by_wl(C.target_type_leaf, 'refl', wl=value, base_path=sample_dir)
            vals.append(DU.get_rend_as_mean(p))

        reflectance.append(vals)

    plotter.plot_reflectance_lab(HSV_value=HSV_values, reflectance=reflectance, powers=powers, plot_name='diffuce_reflectance')

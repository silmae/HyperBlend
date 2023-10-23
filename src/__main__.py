"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in your favourite IDE.
"""

import logging
import os.path
import datetime
import math

import os
import numpy as np
import matplotlib.pyplot as plt

import plotter
import src.leaf_model.nn
from src.leaf_model import surf as SM
from src.forest import forest
from src.blender_scripts import forest_control as control

from src.data import toml_handling as TH, cube_handling as CH, file_names as FN, path_handling as PH
from src.leaf_model import interface as LI
from src import constants as C
from src.reflectance_lab import diffuse_reflectance

from src.rendering import blender_control as BC
from src.gsv import gsv
from src.forest import soil

from src.algae import measurement_spec_24_08_23 as algae
from src.algae import measurement_spec_01_09_23 as M
from src.utils import data_utils as DU
from src.leaf_model import training_data


def write_forest_control(forest_id: str, control_dict: dict):
    TH.write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_forest_scene(forest_id=forest_id), filename='forest_control')


def read_forest_control(forest_id: str) -> dict:
    return TH.read_toml_as_dict(directory=PH.path_directory_forest_scene(forest_id=forest_id), filename='forest_control')


def forest_pipe_test(rng):

    # Generating low resolution random leaves
    set_name = 'low_res_w_dry'
    new_sampling = [450,500,550,600,700,800]
    # LI.generate_prospect_leaf_random(set_name=set_name, leaf_count=2)
    # LI.generate_prospect_leaf(set_name=set_name, sample_id=4) # add one dry default leaf
    # LI.resample_leaf_targets(set_name=set_name, new_sampling=new_sampling)
    # LI.solve_leaf_material_parameters(set_name=set_name, clear_old_results=True)
    #
    leaves = [(set_name, 0, 'Leaf material 1'), (set_name, 1, 'Leaf material 2'), (set_name, 3, 'Leaf material 3')]
    # forest_id = forest.init(leaves=leaves, conf_type='m2m', rng=rng, custom_forest_id='control_test')

    """
    Running forest.init only copies files. Running setup makes the Blender scene renderable.
    """

    forest_id = 'control_test'

    BC.setup_forest(forest_id=forest_id, leaf_name_list=['Leaf material 1', 'Leaf material 2', 'Leaf material 3'])  #, 'Leaf material 4'])

    BC.render_forest(forest_id=forest_id, render_mode='preview')
    BC.render_forest(forest_id=forest_id, render_mode='visibility')
    BC.render_forest(forest_id=forest_id, render_mode='spectral')

    CH.construct_envi_cube(forest_id=forest_id)
    CH.show_cube(forest_id=forest_id)

    # BC.generate_forest_control(global_master=True)



def run_paper_tests():

    nn_name = "lc5_lw1000_b32_lr0.000_split0.10.pt"
    surf_model_name = FN.get_surface_model_save_name('train_iter_4_v4')

    resolution = 5
    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_specchio_nn", copyof="specchio", solver="nn",
                                      solver_model_name=nn_name, plot_resampling=False, use_dumb_sampling=True)
    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_specchio_surf", copyof="specchio", solver="surf",
                                      plot_resampling=False, solver_model_name=surf_model_name, use_dumb_sampling=True)

    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_prospect_nn", copyof="prospect_randoms", solver="nn",
                                      solver_model_name=nn_name, plot_resampling=False, use_dumb_sampling=True)
    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_prospect_surf", copyof="prospect_randoms",
                                      solver="surf", plot_resampling=False, solver_model_name=surf_model_name, use_dumb_sampling=True)


def asym_test(smthng='const_r_var_t'):
    import numpy as np
    from src.leaf_model import leaf_commons as LC
    from src.leaf_model.opt import Optimization
    from src.utils import data_utils

    set_name = f"{smthng}_test"

    n = 10
    const = 0.05
    if smthng == 'const_r_var_t':
        r_list = np.ones((n,)) * const
        t_list = np.linspace(0.1, 0.8, num=n, endpoint=True)
        wls = np.arange(n)
    elif smthng == 'const_t_var_r':
        t_list = np.ones((n,)) * const
        r_list = np.linspace(0.1, 0.8, num=n, endpoint=True)
        wls = np.arange(n)

    data = data_utils.pack_target(wls=wls, refls=r_list, trans=t_list)

    LC.initialize_directories(set_name=set_name, clear_old_results=True)
    TH.write_target(set_name=set_name, data=data)
    # targets = TH.read_target(set_name=set_name, sample_id=0, resampled=False)
    # o = Optimization(set_name=set_name, diffstep=0.01)
    # o.run_optimization(resampled=False, use_threads=True)
    LI.solve_leaf_material_parameters(set_name=set_name, use_dumb_sampling=True, solver='nn', clear_old_results=True, plot_resampling=False)
    print(f"Done {set_name}")


def iterative_train():

    # Iterative train manually
    set_name_iter_1 = "train_iter_1v4_algae"
    # LI.train_models(set_name=set_name_iter_1, generate_data=True, starting_guess_type='curve',
    #                 train_points_per_dim=30, similarity_rt=0.25, train_surf=True, train_nn=False, data_generation_diff_step=0.01)
    set_name_iter_2 = "train_iter_2_v4_algae"
    # surf_model_name = FN.get_surface_model_save_name(set_name_iter_1)
    # LI.train_models(set_name=set_name_iter_2, generate_data=True, starting_guess_type='surf',
    #                 surface_model_name=surf_model_name, similarity_rt=0.5, train_surf=True, train_nn=False,
    #                 train_points_per_dim=50, data_generation_diff_step=0.001)
    set_name_iter_3 = "train_iter_3_v4_algae"
    # surf_model_name = FN.get_surface_model_save_name(set_name_iter_2)
    # LI.train_models(set_name=set_name_iter_3, generate_data=True, starting_guess_type='surf',
    #                 surface_model_name=surf_model_name, similarity_rt=0.75, train_surf=True, train_nn=False,
    #                 train_points_per_dim=70, data_generation_diff_step=0.001)
    set_name_iter_4 = "train_iter_4_v4_algae"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_3)
    LI.train_models(set_name=set_name_iter_4, generate_data=True, starting_guess_type='surf',
                    surface_model_name=surf_model_name, similarity_rt=1.0, train_surf=False, train_nn=True,
                    train_points_per_dim=200, dry_run=False, data_generation_diff_step=0.001, show_plot=True, learning_rate=0.0005)

    # surf_model_name = FN.get_surface_model_save_name(set_name_iter_4)


def algae_leaf(set_name):
    """Solve algae parameters as a leaf (hack so no new code needed)."""

    wls, refl, tran = M.plot_algae(save_thumbnail=True, dont_show=True)
    wls = np.flip(wls)
    refl = np.flip(refl)
    tran = np.flip(tran)
    tran = np.clip(tran,0,1)
    refl = np.clip(refl,0,1)
    data = DU.pack_target(wls=wls,refls=refl,trans=tran)
    TH.write_target(set_name=set_name, data=data)
    LI.solve_leaf_material_parameters(set_name=set_name,use_dumb_sampling=True, resolution=5,
                                      clear_old_results=True, solver='nn')


def reactor_test(algae_leaf_set_name:str,  rng):

    # LI.solve_leaf_material_parameters(set_name=set_name, clear_old_results=True, resolution=5, use_dumb_sampling=True)

    material_name = "Reactor content material"
    algae_leaves = [(algae_leaf_set_name, 0, material_name)]
    # forest_id = forest.init(leaves=algae_leaves, rng=rng, custom_forest_id=algae_scene_id, sun_file_name="lamp_spectra.txt")

    """
    Running forest.init only copies files. Running setup makes the Blender scene renderable.
    """

    BC.setup_forest(forest_id=algae_scene_id, leaf_name_list=[material_name])

    # BC.render_forest(forest_id=forest_id, render_mode='preview')
    # BC.render_forest(forest_id=forest_id, render_mode='visibility')
    # BC.render_forest(forest_id=forest_id, render_mode='spectral')
    #
    # CH.construct_envi_cube(forest_id=forest_id)
    # CH.show_cube(forest_id=forest_id)

    # BC.generate_forest_control(global_master=True)


def make_kettles(light_max_pow=100):
    """

    Ensin Wolfram alphaan ratkastavaksi r_s yhtälöstä: r_g^3 - r_s^3 + (r_g^4 / (n * r_s))
    Sitte lasketaan r_l yhtälöstä: r_g^2 / (n * r_s)
    Lopuksi koodissa voidaan tarkistaa että:
    Vl = 2n pi rl^2 rs
    Vs = Vg + Vl
    Vs = 2 pi rs^3
    Al = 4n pi rl rs

    10 l kattila: r_g = 0.1168
    0.1168^3 - x^3 + (0.1168^4/(4x))
    -> yksi positiivinen juuri r_s = 0.12525
    -> r_l = 0.1168^2 / (4*0.12525) = 0.02723002

    Tarkistus:

    Vl = 2n pi rl^2 rs = 8 pi 0.02723002^2 * 0.12525 = 0.00233407
    Vs = Vg + Vl = 0.010 + 0.00233407 = 0.01233407
    Vs = 2 pi rs^3 = 2 pi 0.12525^3 = 0.0123456 OK
    Al = 4n pi rl rs = 16 pi 0.02723002 * 0.12525 = 0.171433 OK

    100 l kattila: r_g = 0.2515
    0.2515^3 - x^3 + (0.2515^4/(4x))
    r_s = 0.269696
    -> r_l = 0.2515^2 / (4*0.269696) = 0.0619492

    1000 l kattila: r_g = 0.5419
    0.5419^3 - x^3 + (0.5419^4/(8x))
    r_s = 0.562822
    -> r_l = 0.5419^2 / (8*0.562822) = 0.0652195

    :return:
    """

    def calc_V(r):
        V = 2 * math.pi * r**3
        return V

    def calc_A(r):
        A = 4 * math.pi * r**2
        return A

    def check_kettle(r_l, r_s, n, V_g, r_g):
        V_l = 2 * n * math.pi * r_l**2 * r_s
        V_s_check = V_g + V_l
        V_s = 2 * math.pi * r_s**3
        A_l = 4 * n * math.pi * r_l * r_s
        A_g = calc_A(r=r_g)
        V_diff = math.fabs(V_s - V_s_check)
        A_diff = math.fabs(A_l - A_g)
        return V_s, V_diff, A_l, A_diff, V_l

    def calc_r_l(r_g, r_s, n):
        r_l = r_g**2 / (n * r_s)
        return r_l

    def do_kettle_thing(target_vol, r_g, r_s, n, n_rings=1, n1=4, n2=4, n3=4, top_cam_height=2.69728):
        V_g = calc_V(r=r_g)
        A_g = calc_A(r=r_g)
        r_l = calc_r_l(r_g=r_g, r_s=r_s,n=n)
        V_s, V_diff, A_l, A_diff, V_l = check_kettle(r_l=r_l, r_s=r_s, n=n, V_g=V_g, r_g=r_g)
        h_g = 2 * r_g
        h_s = 2 * r_s

        print(f"Target volume {target_vol} l or {target_vol / 1000} m^3: ")
        print(f"\tRadius [m]:      glass = {r_g}, steel {r_s}")
        print(f"\tHeight [m]:      glass = {h_g}, steel {h_s}")
        print(f"\tVolume [m^3]:    glass = {V_g}, steel {V_s} (diff {V_diff})")
        print(f"\tLamp area [m^2]: glass = {A_g}, steel {A_l} (diff {A_diff})")
        print(f"\tRod lamp volume [m^3]: {V_l}")
        print(f"\tRod lamp radius [m]: {r_l}")

        material_name = "Reactor content material"
        algae_leaves = [(algae_leaf_set_name, 0, material_name)]

        # Steel kettle
        forest_id = forest.init(leaves=algae_leaves, rng=rng, custom_forest_id=f"reactor_steel_{target_vol}", sun_file_name="AP67_spectra.txt")
        BC.setup_forest(forest_id=forest_id, leaf_name_list=[material_name], r_kettle=r_s, kettle_type="steel", r_lamp=r_l,
                        n1=n1, n2=n2, n3=n3, n_rings=n_rings, top_cam_height=top_cam_height, light_max_pow=light_max_pow)

        # Glass kettle
        forest_id = forest.init(leaves=algae_leaves, rng=rng, custom_forest_id=f"reactor_glass_{target_vol}", sun_file_name="AP67_spectra.txt")
        BC.setup_forest(forest_id=forest_id, leaf_name_list=[material_name], r_kettle=r_g, kettle_type="glass", top_cam_height=top_cam_height,
                        light_max_pow=light_max_pow)

    # Equation for WA
    # r_g^3 - x^3 + (r_g^4 / (n * x))

    # 10 l kettle:
    # 0.1168^3 - x^3 + (0.1168^4 / (6 * x))
    target_vol = 10
    r_g = 0.1168
    r_s = 0.122677  # from Wolfram Alpha
    n = 6
    top_cam_height = 0.617276
    do_kettle_thing(target_vol=target_vol, r_g=r_g, r_s=r_s, n=n, n1=n, n_rings=1, top_cam_height=top_cam_height)

    # 100 l kettle:
    # 0.2515^3 - x^3 + (0.2515^4 / (12 * x))
    target_vol = 100
    r_g = 0.2515
    r_s = 0.25813  # from Wolfram Alpha
    n = 12
    top_cam_height = 1.26728
    do_kettle_thing(target_vol=target_vol, r_g=r_g, r_s=r_s, n=n, n1=4, n2=8, n_rings=2, top_cam_height=top_cam_height)

    # 1000 l kettle:
    # 0.5419^3 - x^3 + (0.5419^4 / (28 * x))
    target_vol = 1000
    r_g = 0.5419
    r_s = 0.548203 # from Wolfram Alpha
    n = 28
    top_cam_height = 2.69728
    do_kettle_thing(target_vol=target_vol, r_g=r_g, r_s=r_s, n=n, n1=4, n2=8, n3=16, n_rings=3, top_cam_height=top_cam_height)


def render_cubes(light_max_pow):

    scene_id = "reactor_steel_10"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)

    scene_id = "reactor_steel_100"
    BC.render_forest(forest_id="reactor_steel_100", render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)

    scene_id = "reactor_steel_1000"
    BC.render_forest(forest_id="reactor_steel_1000", render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)

    scene_id = "reactor_glass_10"
    BC.render_forest(forest_id="reactor_glass_10", render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)

    scene_id = "reactor_glass_100"
    BC.render_forest(forest_id="reactor_glass_100", render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)

    scene_id = "reactor_glass_1000"
    BC.render_forest(forest_id="reactor_glass_1000", render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)


def show_cubes():

    CH.show_cube(forest_id="reactor_steel_10")
    CH.show_cube(forest_id="reactor_steel_100")
    CH.show_cube(forest_id="reactor_steel_1000")
    CH.show_cube(forest_id="reactor_glass_10")
    CH.show_cube(forest_id="reactor_glass_100")
    CH.show_cube(forest_id="reactor_glass_1000")


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    # logging.basicConfig(stream=sys.stdout, level='INFO')
    path_dir_logs = "../log"
    if not os.path.exists(path_dir_logs):
        os.makedirs(path_dir_logs)

    log_identifier = str(datetime.datetime.now())
    log_identifier = log_identifier.replace(' ', '_')
    log_identifier = log_identifier.replace(':', '')
    log_identifier = log_identifier.replace('.', '')

    log_file_name = f"{log_identifier}.log"
    log_path = PH.join(path_dir_logs, log_file_name)
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='w'),
                            logging.StreamHandler()
                        ])


    rng = np.random.default_rng(4321)

    # Retrain for algae
    # iterative_train()

    # LI.visualize_leaf_models(training_set_name=set_name_iter_1, show_plot=True, plot_surf=False, plot_nn=False)

    ##### ALGAE STUFF #######

    algae_leaf_set_name = "algae_sample_1"
    algae_scene_id = "reactor_steel_10"
    material_name = "Reactor content material"
    light_max_pow = 100 # W / m^2

    # Solve algae as a leaf
    # algae_leaf(set_name=algae_leaf_set_name)

    # These 3 lines do everything at once
    # make_kettles()
    # render_cubes(light_max_pow=light_max_pow)
    show_cubes()

    # BC.render_forest(forest_id="reactor_glass_100",render_mode='top', light_max_pow=light_max_pow)
    # CH.construct_envi_cube(forest_id="reactor_glass_100", light_max_power=light_max_pow)
    # CH.show_cube(forest_id="reactor_glass_100")


    # CH.construct_envi_cube(forest_id=algae_scene_id, light_max_power=100)
    # CH.show_cube(forest_id=algae_scene_id)

    # material_name = "Reactor content material"


    # reactor_test(algae_leaf_set_name=algae_leaf_set_name, rng=rng)


    # BC.setup_forest(forest_id=algae_scene_id, leaf_name_list=[material_name])


    #########################

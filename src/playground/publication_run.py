from data import toml_handling as TH, path_handling as PH, file_names as FN
from slab_model import interface as LI


def write_forest_control(forest_id: str, control_dict: dict):
    TH.write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_system_simulation(forest_id=forest_id), filename='forest_control')


def read_forest_control(forest_id: str) -> dict:
    return TH.read_toml_as_dict(directory=PH.path_directory_system_simulation(forest_id=forest_id), filename='forest_control')


def run_paper_tests():

    nn_name = "lc5_lw1000_b32_lr0.000_split0.10.pt"
    surf_model_name = FN.get_surface_model_save_name('train_iter_4_v4')

    resolution = 5
    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_specchio_nn", copyof="specchio", solver="nn",
                                      solver_dirname=nn_name, plot_resampling=False, use_dumb_sampling=True)
    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_specchio_surf", copyof="specchio", solver="surf",
                                      plot_resampling=False, solver_dirname=surf_model_name, use_dumb_sampling=True)

    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_prospect_nn", copyof="prospect_randoms", solver="nn",
                                      solver_dirname=nn_name, plot_resampling=False, use_dumb_sampling=True)
    LI.solve_leaf_material_parameters(clear_old_results=True, resolution=resolution, set_name="iterative_prospect_surf", copyof="prospect_randoms",
                                      solver="surf", plot_resampling=False, solver_dirname=surf_model_name, use_dumb_sampling=True)


def asym_test(smthng='const_r_var_t'):
    import numpy as np
    from src.slab_model import slab_commons as LC
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

    LC.initialize_directories(slab_sim_name=set_name, clear_old_results=True)
    TH.write_target(set_name=set_name, data=data)
    # targets = TH.read_target(set_name=set_name, sample_id=0, resampled=False)
    # o = Optimization(set_name=set_name, diffstep=0.01)
    # o.run_optimization(resampled=False, use_threads=True)
    LI.solve_leaf_material_parameters(set_name=set_name, use_dumb_sampling=True, solver='nn', clear_old_results=True, plot_resampling=False)
    print(f"Done {set_name}")


def iterative_train():

    # Iterative train manually
    set_name_iter_1 = "train_iter_1v4"
    LI.train_models(set_name=set_name_iter_1, generate_data=True, data_generation_diff_step=0.01,
                    starting_guess_type='curve', similarity_rt=0.25, train_surf=True, train_nn=False,
                    train_points_per_dim=30)
    set_name_iter_2 = "train_iter_2_v4"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_1)
    LI.train_models(set_name=set_name_iter_2, generate_data=True, data_generation_diff_step=0.001,
                    starting_guess_type='surf', similarity_rt=0.5, train_surf=True, train_nn=False,
                    train_points_per_dim=50)
    set_name_iter_3 = "train_iter_3_v4"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_2)
    LI.train_models(set_name=set_name_iter_3, generate_data=True, data_generation_diff_step=0.001,
                    starting_guess_type='surf', similarity_rt=0.75, train_surf=True, train_nn=False,
                    train_points_per_dim=70)
    set_name_iter_4 = "train_iter_4_v4"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_3)
    LI.train_models(set_name=set_name_iter_4, generate_data=False, data_generation_diff_step=0.001,
                    starting_guess_type='surf', similarity_rt=1.0, train_surf=False, train_nn=True,
                    learning_rate=0.0005, train_points_per_dim=200, dry_run=False, show_plot=True)

    # surf_model_name = FN.get_surface_model_save_name(set_name_iter_4)

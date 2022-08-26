"""

Interface for all leaf material related actions.

"""

import numpy as np

import time
import logging

import src.leaf_model.training_data
from src.leaf_model.leaf_commons import  _convert_raw_params_to_renderable, \
    _material_params_to_RT, _build_sample_res_dict, initialize_directories
from src.leaf_model.opt import Optimization
from src.data import file_handling as FH, toml_handling as TH
from src import plotter
from src.leaf_model import nn, surf, leaf_commons as LC


def solve_leaf_material_parameters(set_name: str, resolution=1, solver='nn', clear_old_results=False, nn_name=None,
                                   copyof=None, prospect_parameters: dict = None):


    if prospect_parameters is not None:
        logging.info(f"Prospect parameter generation asked but I don't know how yet.")
        # TODO generate new targets and save to disk with given name

    # TODO copyof some other measurement set
    if copyof:
        FH.copy_target(from_set=copyof, to_set=set_name)
    else:
        initialize_directories(set_name=set_name, clear_old_results=clear_old_results)

    ids = FH.list_target_ids(set_name)
    ids.sort()

    if len(ids) < 1:
        raise RuntimeError(f'Could not find any targets for set "{set_name}".')

    for _, sample_id in enumerate(ids):
        FH.create_opt_folder_structure_for_samples(set_name, sample_id)
        logging.info(f'Starting optimization of sample {sample_id}')
        targets = TH.read_target(set_name, sample_id)

        # Spectral resolution
        if resolution != 1:
            targets = targets[::resolution]

        if solver == 'opt':
            o = Optimization(set_name=set_name)
            o.run_optimization(resolution=resolution)
        elif solver == 'surf' or solver == "nn":
            start = time.perf_counter()

            wls = targets[:, 0]
            r_m = targets[:, 1]
            t_m = targets[:, 2]

            if solver == 'surf':
                ad_raw, sd_raw, ai_raw, mf_raw = surf.predict(r_m=r_m, t_m=t_m)
            elif solver == "nn":
                if nn_name:
                    ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(r_m=r_m, t_m=t_m, nn_name=nn_name)
                else:
                    ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(r_m=r_m, t_m=t_m)

            ad, sd, ai, mf = _convert_raw_params_to_renderable(ad_raw, sd_raw, ai_raw, mf_raw)
            r, t = _material_params_to_RT(set_name, sample_id, wls, ad, sd, ai, mf)

            re = np.abs(r - r_m)
            te = np.abs(t - t_m)
            running_time = (time.perf_counter() - start) / 60.0
            time_process_min = running_time
            time_wall_clock_min = running_time
            sample_result_dict = _build_sample_res_dict(wls, r, r_m, re, t, t_m, te, ad_raw, sd_raw, ai_raw, mf_raw, time_process_min, time_wall_clock_min)

            TH.write_sample_result(set_name, sample_result_dict, sample_id)

            plotter.plot_sample_result(set_name, sample_id, dont_show=True, save_thumbnail=True)
        else:
            raise AttributeError(f"Unknown solver '{solver}'. Use one of ['nn','surf','opt'].")


def train_models(set_name='training_data', show_plot=False, layer_count=9, layer_width=10, epochs=300, batch_size=2,
                 learning_rate=0.001, patience=30, split=0.1, generate_data=False, train_points_per_dim=50):

    if generate_data:
        src.leaf_model.training_data.generate_train_data(set_name=set_name, cuts_per_dim=train_points_per_dim, dry_run=False)

    surf.train(set_name=set_name)
    nn.train(show_plot=show_plot, layer_count=layer_count, layer_width=layer_width, epochs=epochs,
             batch_size=batch_size, learning_rate=learning_rate, patience=patience, split=split, set_name=set_name)

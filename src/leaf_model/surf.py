"""
Surface model stuff

"""

import logging
import math
import numpy as np
from scipy.optimize import curve_fit

import src.leaf_model.training_data as training
from src.data import toml_handling as TH
from src import constants as C
from src.leaf_model.opt import Optimization
from src.leaf_model import surface_functions as FF
from src import plotter
from src.leaf_model import nn
from src.leaf_model import leaf_commons as shared


def predict(r_m, t_m):
    param_dict = TH.read_surface_model_parameters()
    ad_p = param_dict['ad']
    sd_p = param_dict['sd']
    ai_p = param_dict['ai']
    mf_p = param_dict['mf']
    ad_raw = np.clip(FF.function_exp(np.array([r_m, t_m]), *ad_p), 0.0, 1.0)
    sd_raw = np.clip(FF.function_log(np.array([r_m, t_m]), *sd_p), 0.0, 1.0)
    ai_raw = np.clip(FF.function_polynomial(np.array([r_m, t_m]), *ai_p), 0.0, 1.0)
    mf_raw = np.clip(FF.function_exp(np.array([r_m, t_m]), *mf_p), 0.0, 1.0)
    return ad_raw, sd_raw, ai_raw, mf_raw


def train(set_name='training_data'):
    """Train surface model.
    :param set_name:
    """
    _fit_surface(set_name=set_name, show_plot=False, save_params=True)


def _fit_surface(set_name='training_data', show_plot=False, save_params=False, plot_data_as_surface=False, show_nn=True):
    """Fit surfaces.

    Calling this is the same as calling ```train(do_points=False)```.

    TODO fix plotting stuff

    Surface fitting parameters written to disk and plots shown if show_plot=True.
    :param set_name:
    :param plot_data_as_surface:
    :param save_params:
    """
    # ids = FH.list_finished_sample_ids(set_name)
    # for _, sample_id in enumerate(ids):
    result = TH.read_sample_result(set_name, sample_id=0)
    # if show_nn:
    #     rm = np.array(result[C.key_sample_result_rm])
    #     tm = np.array(result[C.key_sample_result_tm])
    #     ad, sd, ai, mf = nn.predict(rm, tm)
    # else:

    ad = np.array(result[C.key_sample_result_ad])
    sd = np.array(result[C.key_sample_result_sd])
    ai = np.array(result[C.key_sample_result_ai])
    mf = np.array(result[C.key_sample_result_mf])

    r  = np.array(result[C.key_sample_result_r])
    t  = np.array(result[C.key_sample_result_t])
    re = np.array(result[C.key_sample_result_re])
    te = np.array(result[C.key_sample_result_te])

    ad, sd, ai, mf, r, t = training.prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=False)

    variable_lol = [ad, sd, ai, mf]
    variable_names = ['ad', 'sd', 'ai', 'mf']

    result_dict = {}

    for i, variable in enumerate(variable_lol):
        # get fit parameters from scipy curve fit
        # https://stackoverflow.com/questions/56439930/how-to-use-the-datasets-to-fit-the-3d-surface

        zlabel = variable_names[i]
        failed = False

        try:
            fittable = FF.function_exp
            if variable_names[i] == 'ai':
                fittable = FF.function_polynomial
            if variable_names[i] == 'sd':
                fittable = FF.function_log
            if variable_names[i] == 'mf':
                fittable = FF.function_exp

            parameters, _ = curve_fit(fittable, [r, t], variable, p0=FF.get_x0())
            result_dict[variable_names[i]] = parameters
            plotter.plot_3d_rt(r, t, variable, zlabel, z_intensity=None, surface_parameters=parameters,
                               fittable=fittable, save_thumbnail=True, show_plot=show_plot,
                               plot_data_as_surface=plot_data_as_surface)

        except RuntimeError as re:
            logging.error(f'Failed to fit for parameter {variable_names[i]}')
            plotter.plot_3d_rt(r, t, variable, zlabel, save_thumbnail=False, show_plot=show_plot)
            failed = True

    if not failed:
        if save_params:
            print(f'Saving surface model parameters')
            TH.write_surface_model_parameters(result_dict)
    else:
        raise RuntimeError(f'Failed to fit all parameters. The result will not be saved.')

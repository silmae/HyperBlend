"""
Surface model stuff

"""

import os
import logging
import numpy as np
from scipy.optimize import curve_fit

import src.leaf_model.training_data as training
from src.data import toml_handling as TH, path_handling as PH
from src import constants as C
from src.leaf_model import surface_functions as FF
from src import plotter


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
        Set name.
    """

    logging.info(f"Starting surface model training.")
    ad, sd, ai, mf, r, t, re, te = training.get_training_data(set_name=set_name)
    ad, sd, ai, mf, r, t = training.prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=False)

    surface_param_dict = {
        'ad': curve_fit(FF.function_exp, [r, t], ad, p0=FF.get_x0())[0],
        'sd': curve_fit(FF.function_log, [r, t], sd, p0=FF.get_x0())[0],
        'ai': curve_fit(FF.function_polynomial, [r, t], ai, p0=FF.get_x0())[0],
        'mf': curve_fit(FF.function_exp, [r, t], mf, p0=FF.get_x0())[0],
    }
    TH.write_surface_model_parameters(surface_param_dict)
    logging.info(f"Surface model training done.")


def exists():
    """Checks whether surface model parameters exist.

    :return:
        Returns True if surface model parameters exist, False otherwise.
    """

    p = PH.path_file_surface_model_parameters()
    return os.path.exists(p)

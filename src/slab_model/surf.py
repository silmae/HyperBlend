"""
Slab simulation surface model implementation.

"""

import os
import logging
import numpy as np
from scipy.optimize import curve_fit

import src.slab_model.training_data as training
from src.data import toml_handling as TH, path_handling as PH, file_names as FN
from src.slab_model import surface_functions as FF


def predict(target_refl, target_tran, surface_model_name: str):
    """Predicts the surface model parameters for given r_m and t_m.
    
    :param target_refl: Target reflectance.
    :param target_tran: Target transmittance.
    :param surface_model_name: Name of the surface model to be used.
    :return: Lists ad, sd, ai, mf (absorption density, scattering density, scattering anisotropy, and mixing factor).
        Use :func:`<slab_commons._convert_raw_params_to_renderable()>` before passing them to rendering method.
    """
    param_dict = TH.read_surface_model_parameters(surface_model_name)
    ad_p = param_dict['ad']
    sd_p = param_dict['sd']
    ai_p = param_dict['ai']
    mf_p = param_dict['mf']
    ad_raw = np.clip(FF.function_exp(np.array([target_refl, target_tran]), *ad_p), 0.0, 1.0)
    sd_raw = np.clip(FF.function_log(np.array([target_refl, target_tran]), *sd_p), 0.0, 1.0)
    ai_raw = np.clip(FF.function_polynomial(np.array([target_refl, target_tran]), *ai_p), 0.0, 1.0)
    mf_raw = np.clip(FF.function_exp(np.array([target_refl, target_tran]), *mf_p), 0.0, 1.0)
    return ad_raw, sd_raw, ai_raw, mf_raw


def train(training_sim_name='training_data'):
    """Train surface model.

    :param training_sim_name:
        Name of the training data slab simulation. Note that the training data actually is another slab
        simulation; just a special kind where we generate the training data points and solved their
        material parameters with the optimization method. No need to change the default name unless you
        generated the data with custom name.
    """

    logging.info(f"Starting surface model training.")
    ad, sd, ai, mf, r, t, re, te = training.get_training_data(training_sim_name=training_sim_name)
    ad, sd, ai, mf, r, t = training.prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=False)

    surface_param_dict = {
        'ad': curve_fit(FF.function_exp, [r, t], ad, p0=FF.get_x0())[0],
        'sd': curve_fit(FF.function_log, [r, t], sd, p0=FF.get_x0())[0],
        'ai': curve_fit(FF.function_polynomial, [r, t], ai, p0=FF.get_x0())[0],
        'mf': curve_fit(FF.function_exp, [r, t], mf, p0=FF.get_x0())[0],
    }

    file_name = FN.get_surface_model_save_name(training_set_name=training_sim_name)
    TH.write_surface_model_parameters(surface_param_dict, file_name=file_name)
    logging.info(f"Surface model training done.")


def exists(file_name=None):
    """Checks whether surface model parameters exist.

    :param file_name:
    :return:
        Returns True if surface model parameters exist, False otherwise.
    """

    p = PH.path_file_surface_model_parameters(file_name=file_name)
    return os.path.exists(p)

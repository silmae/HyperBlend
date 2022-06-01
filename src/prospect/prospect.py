"""
Little PROSPECT interface with some quality-of-life calls.
"""

import numpy as np

from src.prospect import prospect_d as PD


p_range_dict = {
    "n_range"       : (0.8, 2.5), # PROSPECT N parameter [unitless]
    "ab_range"      : (0.0, 80.0), # chlorophyll a + b concentration [ug / cm^2]
    "ar_range"      : (0.0, 20.0), # cartenoid content [ug / cm^2]
    "brown_range"   : (0.0, 1.0), # brown pigment [unitless]
    "w_range"       : (0.0, 0.05), # equivalent water thickness [cm]
    "m_range"       : (0.0, 0.02), # dry mater content [g / cm^2]
    "ant_range"     : (0.0, 40.), # anthocyanin content [ug / cm^2] TODO check proper range
}


p_default_dict = {
    "n"         : 1.5,
    "ab"        : 32.,
    "ar"        : 8.,
    "brown"     : 0.,
    "w"         : 0.016,
    "m"         : 0.009,
    "ant"       : 0.0,
}


from src.utils import spectra_utils as SU
from src.optimization import Optimization
import logging
from src.data import file_handling as FH
from src.data import toml_handling as TH
from src.data import path_handling as PH

prospect_set_name = 'prospect_randoms'


def run_prospect_randoms_simulation():
    o = Optimization(set_name=prospect_set_name)
    o.run_optimization(resolution=5, prediction_method='nn')


def make_random_leaf_targets(count=1, resolution=5):

    FH.create_first_level_folders(prospect_set_name) # make sure to have directories created

    for i in range(count):
        wls, r, t, p_dict = run_prospect_random()
        logging.info(f"Generating random leaf data with prospect.")
        SU._make_target(prospect_set_name, wls=wls, r_m=r, t_m=t, sample_id=i) # sample directories are now created
        dict_dir = PH.path_directory_sample(prospect_set_name, sample_id=i)
        dict_name = f'prospect_params_{i}'
        TH.write_dict_as_toml(p_dict, directory=dict_dir, filename=dict_name)



def run_prospect(n, ab, ar, brown, w, m, ant):
    """

    :param n:
        PROSPECT N parameter [unitless]
    :param ab:
        chlorophyll a + b concentration [ug / cm^2]
    :param ar:
        cartenoid content [ug / cm^2]
    :param brown:
        brown pigment [unitless]
    :param w:
        equivalent water thickness [cm]
    :param m:
        dry mater content [g / cm^2]
    :param ant:
        anthocyanin content [ug / cm^2] TODO check proper range
    :return:
    """

    p_dict = get_prospect_params(n, ab, ar, brown, w, m, ant)
    wls, r, t = run_prospect_with_dict(p_dict)
    return wls, r, t


def run_prospect_random():

    def get_val(center, range):
        div = 10
        res = np.random.normal(loc=center, scale=(range[1] - range[0]) / div)
        res = np.clip(res, range[0], range[1])
        return res

    p_dict = get_prospect_params(
        n       = get_val(p_default_dict["n"], p_range_dict["n_range"]),
        ab      = get_val(p_default_dict["ab"], p_range_dict["ab_range"]),
        ar      = get_val(p_default_dict["ar"], p_range_dict["ar_range"]),
        brown   = get_val(p_default_dict["brown"], p_range_dict["brown_range"]),
        w       = get_val(p_default_dict["w"], p_range_dict["w_range"]),
        m       = get_val(p_default_dict["m"], p_range_dict["m_range"]),
        ant     = get_val(p_default_dict["ant"], p_range_dict["ant_range"]),
    )
    wls, r, t = run_prospect_with_dict(p_dict)
    return wls, r, t, p_dict


def run_prospect_with_dict(prospect_params:dict):
    wls, r, t = PD.run_prospect(
        n=prospect_params["n"],
        cab=prospect_params["ab"],
        car=prospect_params["ar"],
        cbrown=prospect_params["brown"],
        cw=prospect_params["w"],
        cm=prospect_params["m"],
        ant=prospect_params["ant"],
        nr=None, kab=None, kcar=None, kbrown=None, kw=None,
        km=None, kant=None, alpha=40.)
    return wls, r, t


def get_prospect_params(n, ab, ar, brown, w, m, ant):
    p_dict = {
        "n": n,
        "ab": ab,
        "ar": ar,
        "brown": brown,
        "w": w,
        "m": m,
        "ant": ant,
    }
    return p_dict


def p_dict_to_hash(p_dict):
    hash = "".join(list([f"{val:.0f}" for key,val in p_dict.items()]))
    return hash


def get_default_prospect_leaf():
    wls, r, t = run_prospect_with_dict(p_default_dict)
    return wls,r,t


def get_default_prospect_leaf_dry():
    wls, r, t = PD.run_prospect(
        n=n_default,
        cab=cab_default,
        car=car_default,
        cbrown=cbrown_default,
        cw=cw_default * 0.1,
        cm=cm_default,
        ant=ant_default,
        nr=None, kab=None, kcar=None, kbrown=None, kw=None,
        km=None, kant=None, alpha=40.)
    return wls, r, t

"""
Little PROSPECT interface with some quality-of-life calls.
"""

import logging
import numpy as np

from src.prospect import prospect_d as PD
from src.utils import spectra_utils as SU
from src.data import file_handling as FH
from src.data import toml_handling as TH
from src.data import path_handling as PH


def make_random_leaf_targets(set_name, count=1):
    """ Generate count number of random PROSPECT leaves.

    :param set_name:
        Set name to be used.
    :param count:
        How many target leaves are generated to the set.
    """

    FH.create_first_level_folders(set_name) # make sure to have directories created

    for i in range(count):
        wls, r, t, p_dict = run_prospect_random()
        logging.info(f"Generating random leaf data with prospect.")
        SU._make_target(set_name, wls=wls, r_m=r, t_m=t, sample_id=i) # sample directories are now created
        dict_dir = PH.path_directory_sample(set_name, sample_id=i)
        dict_name = f'prospect_params_{i}'
        TH.write_dict_as_toml(p_dict, directory=dict_dir, filename=dict_name)


def make_leaf_target(set_name, sample_id=0, n=None, ab=None, ar=None, brown=None, w=None, m=None, ant=None):
    """ Run prospect simulation with given PROSPECT parameters.

    If any of the values are not provided, default values are used.
    You get the default PROSPECT leaf by calling without any arguments.

    :param set_name:
        Set name where the target is saved.
    :param sample_id:
        Sample id for this target. Default is 0. Overwrites existing targets if existing id is given.
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
        anthocyanin content [ug / cm^2]
    :return:
        Tuple (wls, r, t) and writes the target to the disk.
    """

    FH.create_first_level_folders(set_name) # make sure to have directories created

    if n is None:
        n = p_default_dict["n"]
    if ab is None:
        ab = p_default_dict["ab"]
    if ar is None:
        ar = p_default_dict["ar"]
    if brown is None:
        brown = p_default_dict["brown"]
    if w is None:
        w = p_default_dict["w"]
    if m is None:
        m = p_default_dict["m"]
    if ant is None:
        ant = p_default_dict["ant"]

    p_dict = prospect_params_as_dict(n, ab, ar, brown, w, m, ant)
    wls, r, t = run_prospect_with_dict(p_dict)

    logging.info(f"Generating random leaf data with prospect.")
    SU._make_target(set_name, wls=wls, r_m=r, t_m=t, sample_id=sample_id)  # sample directory is now created
    dict_dir = PH.path_directory_sample(set_name, sample_id=sample_id)
    dict_name = f'prospect_params_{sample_id}' # save used prospect parameters
    TH.write_dict_as_toml(p_dict, directory=dict_dir, filename=dict_name)
    return wls, r, t


def run_prospect_random():
    """ Run PROSPECT with random parameters.

    PROSPECT parameters are randomly drawn from normal distribution
    centered at default PRSOPECT leaf parameters.

    :return:
        (wls, r, t, p_dict), where p_dict is a dictionary of used random PROSPECT parameters.
    """

    def get_val(center, range):
        div = 10
        res = np.random.normal(loc=center, scale=(range[1] - range[0]) / div)
        res = np.clip(res, range[0], range[1])
        return res

    p_dict = prospect_params_as_dict(
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
    """ Run PROSPECT simulation with a parameter dictionary.

    You can get the dictionary by calling prospect_params_as_dict()
    with desired parameter values.

    :param prospect_params:
        Dictionary of PROSPECT parameters as returned by prospect_params_as_dict().
    :return:
        Tuple (wls,r,t)
    """
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


def prospect_params_as_dict(n, ab, ar, brown, w, m, ant):
    """ Turn set of PROSPECT parameters into a dictionary.

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
        anthocyanin content [ug / cm^2]
    :return:
        Dictionary with same key names as in parameters.
    """

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
    """Turn PROSPECT parameter dictionary into a hash-kind of string.

    Can be used for naming files.
    """

    phash = "".join(list([f"{val:.0f}" for key,val in p_dict.items()]))
    return phash


def get_default_prospect_leaf():
    """Run PROSPECT with default parameters.

    :return:
        Tuple (wls,r,t)
    """
    wls, r, t = run_prospect_with_dict(p_default_dict)
    return wls,r,t


p_range_dict = {
    "n_range"       : (0.8, 2.5), # PROSPECT N parameter [unitless]
    "ab_range"      : (0.0, 80.0), # chlorophyll a + b concentration [ug / cm^2]
    "ar_range"      : (0.0, 20.0), # cartenoid content [ug / cm^2]
    "brown_range"   : (0.0, 1.0), # brown pigment [unitless]
    "w_range"       : (0.0, 0.05), # equivalent water thickness [cm]
    "m_range"       : (0.0, 0.02), # dry mater content [g / cm^2]
    "ant_range"     : (0.0, 40.), # anthocyanin content [ug / cm^2]
}
"""Clipping range for PROSPECT parameters. Used in when generating random leaves."""

p_default_dict = {
    "n"         : 1.5,
    "ab"        : 32.,
    "ar"        : 8.,
    "brown"     : 0.,
    "w"         : 0.016,
    "m"         : 0.009,
    "ant"       : 0.0,
}
"""Default PROSPECT parameters. These are used for generating random leaves."""

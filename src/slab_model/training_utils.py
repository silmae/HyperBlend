"""

Some common training utilities for the slab model for avoiding circular imports.

"""

import logging

import numpy as np

from src import constants as C
from src.data import toml_handling as TH, toml_handling as T


def prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=False):
    """Prune bad datapoints from training data.

    Data point is considered bad if either reflectance or transmittance error is
    more than 1%.

    :param ad:
        Numpy array absorption particle density.
    :param sd:
        Numpy array scattering particle density.
    :param ai:
        Numpy array scattering anisotropy.
    :param mf:
        Numpy array mix factor.
    :param r:
        Numpy array reflectance.
    :param t:
        Numpy array transmittance.
    :param re:
        Numpy array reflectance error.
    :param te:
        Numpy array transmittance error.
    :param invereted:
        If true, instead of good points, the bad points will be returned.
    :return:
        Pruned ad,sd,ai,mf,r,t corresponding to arguments.
    """

    max_error = 0.02  # 1%
    logging.info(
        f"Points with error of reflectance or transmittance greater than '{max_error}' will be pruned."
    )

    to_delete = [(a > max_error or b > max_error) for a, b in zip(re, te)]

    if invereted:
        to_delete = np.invert(to_delete)

    initial_count = len(ad)
    logging.info(f"Initial point count {initial_count} in training data.")

    to_delete = np.where(to_delete)[0]
    ad = np.delete(ad, to_delete)
    sd = np.delete(sd, to_delete)
    ai = np.delete(ai, to_delete)
    mf = np.delete(mf, to_delete)
    r = np.delete(r, to_delete)
    t = np.delete(t, to_delete)

    bad_points_count = initial_count - len(ad)

    if not invereted:
        logging.info(
            f"Pruned {len(to_delete)} ({(bad_points_count/initial_count)*100:.2}%) points because exceeding error threshold {max_error}."
        )
        logging.info(f"Point count after pruning {len(ad)}.")

    return ad, sd, ai, mf, r, t


def get_training_data(training_sim_name: str):
    """Returns training data.

    :param training_sim_name:
        Name of the training data slab simulation. Note that the training data actually is another slab
        simulation; just a special kind where we generate the training data points and solved their
        material parameters with the optimization method.
    :return:
        Returns ad, sd, ai, mf, r, t, re, te Numpy arrays (vector).
    """

    if training_sim_name is None or not isinstance(training_sim_name, str):
        raise AttributeError("Training simulation name must be provided.")

    result = TH.read_signal_result(training_sim_name, signal_id=0)
    ad = np.array(result[C.key_sample_result_ad])
    sd = np.array(result[C.key_sample_result_sd])
    ai = np.array(result[C.key_sample_result_ai])
    mf = np.array(result[C.key_sample_result_mf])

    r = np.array(result[C.key_sample_result_r])
    t = np.array(result[C.key_sample_result_t])
    re = np.array(result[C.key_sample_result_re])
    te = np.array(result[C.key_sample_result_te])
    return ad, sd, ai, mf, r, t, re, te


def get_starting_guess_points(set_name: str = None):
    """Get starting guess points.

    NOTE: Points where reflectance or transmittance error exceeds 0.002 are deleted.

    :param set_name:
        Custom set name to fetch the data from. If not given, default set name variable
        'starting_guess_set_name' stored in constants.py is used.
    :return:
        a_list, ad_list, sd_list, ai_list, mf_list
    """

    if set_name is None:
        set_name = C.starting_guess_set_name

    result_dict = T.read_signal_result(set_name, 0)
    wls = result_dict[C.key_sample_result_wls]

    re_list = np.array(
        [r for _, r in sorted(zip(wls, result_dict[C.key_sample_result_re]))]
    )
    te_list = np.array(
        [t for _, t in sorted(zip(wls, result_dict[C.key_sample_result_te]))]
    )
    eps = 0.002

    r_list = np.array(
        [r for _, r in sorted(zip(wls, result_dict[C.key_sample_result_r]))]
    )
    t_list = np.array(
        [t for _, t in sorted(zip(wls, result_dict[C.key_sample_result_t]))]
    )
    ad_list = np.array(
        [ad for _, ad in sorted(zip(wls, result_dict[C.key_sample_result_ad]))]
    )
    sd_list = np.array(
        [sd for _, sd in sorted(zip(wls, result_dict[C.key_sample_result_sd]))]
    )
    ai_list = np.array(
        [ai for _, ai in sorted(zip(wls, result_dict[C.key_sample_result_ai]))]
    )
    mf_list = np.array(
        [mf for _, mf in sorted(zip(wls, result_dict[C.key_sample_result_mf]))]
    )

    r_list = r_list[(re_list < eps) & (te_list < eps)]
    t_list = t_list[(re_list < eps) & (te_list < eps)]
    ad_list = ad_list[(re_list < eps) & (te_list < eps)]
    sd_list = sd_list[(re_list < eps) & (te_list < eps)]
    ai_list = ai_list[(re_list < eps) & (te_list < eps)]
    mf_list = mf_list[(re_list < eps) & (te_list < eps)]

    a_list = np.ones_like(r_list) - (r_list + t_list)  # modeled absorptions 1 - (r+t)

    return a_list, ad_list, sd_list, ai_list, mf_list

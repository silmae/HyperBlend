"""
Functionality regarding training data generation for
surface model and neural network training.

This is somewhat specific functionality so it is not included
in the leaf model interface script.
"""

import logging
import math

import numpy as np
from matplotlib import pyplot as plt

from src import constants as C, plotter
from src.data import toml_handling as TH, file_handling as FH
from src.leaf_model.opt import Optimization


"""
Additional conditions after analyzing erronous areas. 
Constants for two lines equation k*r + b that cut the 
edges of data set away. 
"""
k1 = 3.8
k2 = 0.5
b1 = 0.02
b2 = -0.035


def visualize_training_data_pruning(set_name="training_data", show=False, save=True):
    """Visualizes training data. Can be saved to disk or shown directly (or both).

    :param set_name:
        Name of the training data set. Change only if custom name was used in data generation.
    :param show:
        Show interactive plot to user. Default is ```False```.
    :param save:
        Save plot to disk. Default is ```True```.
    """

    result = TH.read_sample_result(set_name, sample_id=0)
    ad = np.array(result[C.key_sample_result_ad])
    sd = np.array(result[C.key_sample_result_sd])
    ai = np.array(result[C.key_sample_result_ai])
    mf = np.array(result[C.key_sample_result_mf])
    r = np.array(result[C.key_sample_result_rm])
    t = np.array(result[C.key_sample_result_tm])
    re = np.array(result[C.key_sample_result_re])
    te = np.array(result[C.key_sample_result_te])
    _, _, _, _, r_bad, t_bad = prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=True)
    _, _, _, _, r_good, t_good = prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=False)
    plotter.plot_training_data_set(r_good=r_good, r_bad=r_bad, t_good=t_good, t_bad=t_bad,
                                   k1=k1, b1=b1, k2=k2, b2=b2, show=show, save=save)


def prune_training_data(ad, sd, ai, mf, r, t, re, te, invereted=False):
    """ Prune bad datapoints from training data.

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

    max_error = 0.01 # 1%
    logging.info(f"Points with error of reflectance or transmittance greater than '{max_error}' will be pruned.")

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
        logging.info(f"Pruned {len(to_delete)} ({(bad_points_count/initial_count)*100:.2}%) points because exceeding error threshold {max_error}.")
        logging.info(f"Point count after pruning {len(ad)}.")

    return ad, sd, ai, mf, r, t


def generate_train_data(set_name='training_data', dry_run=True, cuts_per_dim=10, maxdiff_rt=0.25):
    """Generate reflectance-transmittance pairs as training data for surface fitting and neural network.

    Generated data will have fake wavelengths attached to them. They run from 1 to the number of
    generated points.

    If ``dry_run=True``, only pretends to generate the points. This is useful for testing how
    different ``cuts_per_dim`` values affect the actual point count.

    Data visualization is saved to disk when the data has been generated.

    :param set_name:
        Optionally change the ``set_name`` that is used for destination directory. If other
        than default is used, it must be taken into account when training, i.e., pass the same
        name for training method.
    :param dry_run:
        If true, just prints how many points would have been generated. Note that it
        is not the same as ``cuts_per_dim`` ^2 because parts of the space are not
        usable and will be cut out.
    :param cuts_per_dim:
        Into how many parts each dimension (R,T) are cut in interval [0,1].
    :param maxdiff_rt:
        Controls the symmetry of generated pairs, i.e., how much each R value can differ from
        respective T value. Using greater than 0.25 will cause generating a lot of points
        that will fail to be optimized properly (and will be pruned before training).
    """

    FH.create_first_level_folders(set_name)

    data = []
    fake_wl = 1  # Set dummy wavelengths so that the rest of the code is ok with the files
    R = np.linspace(0, 1.0, cuts_per_dim, endpoint=True)
    T = np.linspace(0, 1.0, cuts_per_dim, endpoint=True)
    for i, r in enumerate(R):
        for j, t in enumerate(T):
            # Do not allow r+t to exceed 1 as it would break conservation of energy
            if not r + t < 1.0:
                continue
            # ensure some amount of symmetry
            if math.fabs(r - t) > maxdiff_rt:
                continue

            # Cutoff points where R and T are low and dissimilar as they will fail anyway.
            if t > r * k1 + b1:
                continue
            if t < r * k2 + b2:
                continue

            wlrt = [fake_wl, r, t]
            data.append(wlrt)
            fake_wl += 1

    if not dry_run:
        logging.info(f"Generated {len(data)} evenly spaced reflectance transmittance targets.")
        TH.write_target(set_name, data, sample_id=0)
        o = Optimization(set_name=set_name)
        o.run_optimization()
    else:
        logging.info(f"Would have generated {len(data)} evenly spaced reflectance transmittance pairs"
                     f"but this was just a dry run..")

    visualize_training_data_pruning(set_name=set_name, show=False, save=True)

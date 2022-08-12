"""
Shared functionality that is used in the surface fitting model and in neural network model.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

from src.data import toml_handling as TH
from src import constants as C



def visualize_training_data_pruning(set_name="surface_train"):

        result = TH.read_sample_result(set_name, sample_id=0)
        ad = np.array(result[C.key_sample_result_ad])
        sd = np.array(result[C.key_sample_result_sd])
        ai = np.array(result[C.key_sample_result_ai])
        mf = np.array(result[C.key_sample_result_mf])
        r = np.array(result[C.key_sample_result_rm])
        t = np.array(result[C.key_sample_result_tm])
        re = np.array(result[C.key_sample_result_re])
        te = np.array(result[C.key_sample_result_te])
        _, _, _, _, r_bad, t_bad = get_training_data(ad, sd, ai, mf, r, t, re, te, pruned=False)
        _, _, _, _, r_good, t_good = get_training_data(ad, sd, ai, mf, r, t, re, te, pruned=True)
        plt.scatter(r_good,t_good,c='b', alpha=0.5, marker='.')
        plt.scatter(r_bad,t_bad,c='r', alpha=0.5, marker='.')
        plt.plot([0,0.6],[0,0.4], c='black', linewidth=3)
        plt.plot([0,0.7],[0,0.7], c='black', linewidth=3)
        # plt.plot([0,0.02],[0.09,0.033], c='black', linewidth=3)
        k1 = 3
        k2 = 0.5
        plt.plot([0,0.09],[0.02,0.33], c='black', linewidth=3)
        # plt.plot([0.05,0.0],[0.4,0.18], c='black', linewidth=3)
        plt.plot([0.05,0.45],[0.0,0.2], c='black', linewidth=3)
        plt.xlabel('R')
        plt.ylabel('T')
        plt.show()



def get_training_data(ad,sd,ai,mf,r,t,re,te,pruned=True):

        set_name = "surface_train"
        max_error = 0.01
        low_cut = 0.0

        logging.info(f"Fetching training data from set '{set_name}'.")
        logging.info(f"Points with error of reflectance or transmittance greater than '{max_error}' will be pruned.")

        bad = [(a > max_error or b > max_error) for a, b in zip(re, te)]
        # bad = [(a > max_error) for a, b in zip(re, te)] # errors of reflectance
        # bad = [(b > max_error) for a, b in zip(re, te)] # errors of transmittance
        low_cut = [(a < low_cut or b < low_cut) for a, b in zip(r, t)]
        to_delete = np.logical_or(bad, low_cut)
        # to_delete = low_cut


        if not pruned:
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

        logging.info(f"Pruned {len(to_delete)} ({(bad_points_count/initial_count)*100:.2}%) points because exceeding error threshold {max_error}.")
        logging.info(f"Point count after pruning {len(ad)}.")

        return ad,sd,ai,mf,r,t


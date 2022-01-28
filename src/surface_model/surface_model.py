"""
Surface model stuff

"""

import logging
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from src.data import toml_handling as TH
from src import constants as C
from src.optimization import Optimization
from src.surface_model import fitting_function as FF

set_name = 'surface_train'


def train():
    """Train surface model."""

    generate_train_data(num_points=10)
    o = Optimization(set_name)
    o.run_optimization(prediction_method='optimization')
    fit_surface(show_plot=False)


def generate_train_data(num_points=10):
    """Generate reflectance-transmittance pairs for surface fitting.

    Generated data is saved to disk to be used in optimization.
    """

    data = []
    fake_wl = 1 # Set dummy wavelengths so that the rest of the code is ok with the files
    for i,r in enumerate(np.linspace(0, 0.6, num_points, endpoint=True)):
        for j,t in enumerate(np.linspace(0, 0.6, num_points, endpoint=True)):
            # Do not allow r+t to exceed 1 as it would break conservation of energy
            if r + t > 1.:
                continue
            # ensure some amount of symmetry
            if math.fabs(r-t) > 0.2:
                continue

            wlrt = [fake_wl, r, t]
            data.append(wlrt)
            fake_wl += 1
    logging.info(f"Generated {len(data)} evenly spaced reflectance transmittance pairs.")
    TH.write_target(set_name, data, sample_id=0)


def fit_surface(show_plot=False):
    """Fit surfaces.

    Surface fitting parameters written to disk and plots shown if show_plot=True.
    """
    # ids = FH.list_finished_sample_ids(set_name)
    # for _, sample_id in enumerate(ids):
    result = TH.read_sample_result(set_name, sample_id=0)
    ad = np.array(result[C.key_sample_result_ad])
    sd = np.array(result[C.key_sample_result_sd])
    ai = np.array(result[C.key_sample_result_ai])
    mf = np.array(result[C.key_sample_result_mf])
    r  = np.array(result[C.key_sample_result_r])
    t  = np.array(result[C.key_sample_result_t])
    re = np.array(result[C.key_sample_result_re])
    te = np.array(result[C.key_sample_result_te])

    max_error = 0.015
    bad = [(a > max_error or b > max_error) for a,b in zip(re, te)]
    bad = np.where(bad)[0]
    np.delete(ad, bad)
    np.delete(sd, bad)
    np.delete(ai, bad)
    np.delete(mf, bad)
    np.delete(r, bad)
    np.delete(t, bad)

    variable_lol = [ad, sd, ai, mf]
    variable_names = ['ad', 'sd', 'ai', 'mf']

    result_dict = {}

    for i, variable in enumerate(variable_lol):
        # get fit parameters from scipy curve fit
        # https://stackoverflow.com/questions/56439930/how-to-use-the-datasets-to-fit-the-3d-surface
        parameters, _ = curve_fit(FF.function, [r, t], variable)
        result_dict[variable_names[i]] = parameters
        if show_plot:
            num_points = 25
            model_x_data = np.linspace(min(r), max(r), num_points)
            model_y_data = np.linspace(min(t), max(t), num_points)
            # create coordinate arrays for vectorized evaluations
            R, T = np.meshgrid(model_x_data, model_y_data)
            # calculate Z coordinate array
            Z = FF.function(np.array([R, T]), *parameters)
            # setup figure object
            fig = plt.figure()
            # setup 3d object
            ax = plt.axes(projection="3d")
            # plot surface
            ax.plot_surface(R, T, Z, alpha=0.5)
            # plot input data
            ax.scatter(r, t, variable, color='red')
            # set plot descriptions
            ax.set_xlabel('R')
            ax.set_ylabel('T')
            ax.set_zlabel(variable_names[i])
            plt.show()

    TH.write_surface_model_parameters(result_dict)

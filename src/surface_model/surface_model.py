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


def train(do_points=True, num_points=50):
    """Train surface model."""
    if do_points:
        generate_train_data(num_points)
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
            if r + t >= 0.999999:
                continue
            # ensure some amount of symmetry
            if math.fabs(r-t) > 0.1:
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

    max_error = 0.003
    low_cut = 0.0
    bad = [(a > max_error or b > max_error) for a,b in zip(re, te)]
    # bad = np.where(bad)[0]
    low_cut = [(a < low_cut or b < low_cut) for a,b in zip(r, t)]
    to_delete = np.logical_or(bad, low_cut)
    # to_delete = bad

    to_delete = np.where(to_delete)[0]
    ad = np.delete(ad, to_delete)
    sd = np.delete(sd, to_delete)
    ai = np.delete(ai, to_delete)
    mf = np.delete(mf, to_delete)
    r  = np.delete(r , to_delete)
    t  = np.delete(t , to_delete)

    variable_lol = [ad, sd, ai, mf]
    variable_names = ['ad', 'sd', 'ai', 'mf']

    result_dict = {}

    for i, variable in enumerate(variable_lol):
        # get fit parameters from scipy curve fit
        # https://stackoverflow.com/questions/56439930/how-to-use-the-datasets-to-fit-the-3d-surface

        num_points = 25
        model_x_data = np.linspace(0, max(r), num_points)
        model_y_data = np.linspace(0, max(t), num_points)
        # setup figure object
        fig = plt.figure()
        # setup 3d object
        ax = plt.axes(projection="3d")
        ax.set_xlabel('R')
        ax.set_ylabel('T')
        ax.set_zlabel(variable_names[i])
        failed = False

        try:
            fittable = FF.function_exp
            if variable_names[i] == 'ai':
                fittable = FF.function2
            if variable_names[i] == 'sd':
                fittable = FF.function_log
            if variable_names[i] == 'mf':
                fittable = FF.function_free

            parameters, _ = curve_fit(fittable, [r, t], variable, p0=FF.get_x0())
            result_dict[variable_names[i]] = parameters
            if show_plot:
                # create coordinate arrays for vectorized evaluations
                R, T = np.meshgrid(model_x_data, model_y_data)
                # calculate Z coordinate array
                Z = fittable(np.array([R, T]), *parameters)
                # plot input data
                ax.scatter(r, t, variable, c=variable, cmap=plt.cm.hot)
                # plot surface
                ax.plot_surface(R, T, Z, alpha=0.5)
                plt.show()
        except RuntimeError as re:
            logging.error(f'Failed to fit for parameter {variable_names[i]}')
            if show_plot:
                # plot input data
                ax.scatter(r, t, variable, color='red')
                plt.show()
                failed = True
            # raise

        # if show_plot:
        #         # set plot descriptions
        #         ax.set_xlabel('R')
        #         ax.set_ylabel('T')
        #         ax.set_zlabel(variable_names[i])
        #         plt.show()

    if not failed:
        TH.write_surface_model_parameters(result_dict)
    else:
        raise RuntimeError(f'Failed to fit all parameters. The result will not be saved.')

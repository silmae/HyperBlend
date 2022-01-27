"""
Entry point of the program.

There is no user interface to the program (yet) so code your calls here
and run in IDE.
"""

import logging

import sys
import numpy as np

from src import presets
from src.optimization import Optimization
from data import toml_handling as TH
from src.utils import general_utils as GU

def fifi(max_degree = 30):
    residuals = []
    min_residual = 1e30
    best_coef = []
    best_i = 0
    wls, r, _ = get_default_P_leaf()
    for i in range(max_degree):
        series, residue = fit(wls, r, degree=i)
        residuals.append(residue)
        print(f'residue: {residue}')
        if residue < min_residual:
            min_residual = residue
            coeff = series.convert().coef
            best_i = i

        # fig, ax = plt.subplots(ncols=2)
        # ax[0].plot(range(max_degree), residuals)
        # ax[0].scatter(best_i, residuals[i], color='red')

        # ax[1].plot(wls, r, color='red')
        # y = np.array([np.sum(np.array([coeff[i] * (j ** i) for i in range(len(coeff))])) for j in wls])
        poly = np.poly1d(series)
        new_x = np.linspace(wls[0], wls[-1])
        new_y = poly(wls)
        # plt.plot(wls, new_y, color='blue')
        plt.plot(wls, r, ".", wls, new_y(wls))
        plt.show()


def fit(wls, r, degree):
    from numpy.polynomial import Polynomial
    series, stuff = Polynomial.fit(wls, r, deg=degree, domain=[0,1], full=True)
    resi = stuff[0]
    if resi.size == 1:
        resi = resi[0]
    else:
        resi = 0
    return series, resi


def get_default_P_leaf():
    wls, r, t = PD.run_prospect(
        n=1.5,
        cab=32,
        car=8,
        cbrown=0.,
        cw=0.016,
        cm=0.009,
        ant=0.0,
        nr=None, kab=None, kcar=None, kbrown=None, kw=None,
        km=None, kant=None, alpha=40.)
    return wls,r,t


def function(data, a, b, c):
    r = data[0]
    t = data[1]
    res = a * (r**b) * (t**c)
    return res


def get_param_path(set_name):
    bp = PH.path_directory_set(set_name)
    fn = 'suface_fit.toml'
    p = PH.join(bp, fn)
    return p


def fit_surface(set_name, show_plot=False):
    # ids = FH.list_finished_sample_ids(set_name)
    # for _, sample_id in enumerate(ids):
    result = TH.read_sample_result(set_name, sample_id=0)
    ad = result[C.key_sample_result_ad]
    sd = result[C.key_sample_result_sd]
    ai = result[C.key_sample_result_ai]
    mf = result[C.key_sample_result_mf]
    r_m = result[C.key_sample_result_rm]
    t_m = result[C.key_sample_result_tm]
    variable_lol = [ad, sd, ai, mf]
    variable_names = ['ad', 'sd', 'ai', 'mf']

    p = get_param_path(set_name)
    result_dict = {}

    for i, variable in enumerate(variable_lol):
        # get fit parameters from scipy curve fit
        # https://stackoverflow.com/questions/56439930/how-to-use-the-datasets-to-fit-the-3d-surface
        parameters, _ = curve_fit(function, [r_m, t_m], variable)
        result_dict[variable_names[i]] = parameters
        if show_plot:
            num_points = 25
            model_x_data = np.linspace(min(r_m), max(r_m), num_points)
            model_y_data = np.linspace(min(t_m), max(t_m), num_points)
            # create coordinate arrays for vectorized evaluations
            R, T = np.meshgrid(model_x_data, model_y_data)
            # calculate Z coordinate array
            Z = function(np.array([R, T]), *parameters)
            # setup figure object
            fig = plt.figure()
            # setup 3d object
            ax = plt.axes(projection="3d")
            # plot surface
            ax.plot_surface(R, T, Z, alpha=0.5)
            # plot input data
            ax.scatter(r_m, t_m, variable, color='red')
            # set plot descriptions
            ax.set_xlabel('R')
            ax.set_ylabel('T')
            ax.set_zlabel(variable_names[i])
            plt.show()

    with open(p, 'w+') as file:
        toml.dump(result_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def get_model_params(set_name):
    p = get_param_path(set_name)
    with open(p, 'r') as file:
        result = toml.load(file)
    return result


def predict(set_name, r_m, t_m, wl):
    ad_p, sd_p, ai_p, mf_p = get_model_params(set_name)
    ad = function(np.array([r_m, t_m]), *ad_p)
    sd = function(np.array([r_m, t_m]), *sd_p)
    ai = function(np.array([r_m, t_m]), *ai_p)
    mf = function(np.array([r_m, t_m]), *mf_p)



if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    from scipy.optimize import curve_fit
    import toml

    import prospect_d as PD
    import plotter as P
    import matplotlib.pyplot as plt
    from src.data import file_handling as FH
    from src.data import toml_handling as TH
    from src import constants as C
    from src.data import path_handling as PH
    from src.utils import spectra_utils as SU


    # set_name = 'specchio_5nm'
    set_name = 'surface_test_predict'
    o = Optimization(set_name)
    # wls,r,t = get_default_P_leaf()
    # SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
    o.run_optimization(resolution=5)

    # fit_surface(set_name, show_plot=False)

    # fig, ax = plt.subplots(ncols=2, nrows=2)
    # ax[0][0].axes(projection='3d')
    # ax[0][1].axes(projection='3d')
    # ax[1][0].axes(projection='3d')
    # ax[1][1].axes(projection='3d')

    # ax[0][0].scatter3D(r_mean, t_mean, adens_mean)
    # ax[0][1].scatter3D(r_mean, t_mean, sdens_mean)
    # ax[1][0].scatter3D(r_mean, t_mean, ai_mean)
    # ax[1][1].scatter3D(r_mean, t_mean, mf_mean)
    # ax[0][0].set_title('adens_mean')
    # ax[0][1].set_title('sdens_mean')
    # ax[1][0].set_title('ai_mean')
    # ax[1][1].set_title('mf_mean')

    # fig = plt.figure()

    # defining the axes with the projection
    # as 3D so as to plot 3D graphs
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(r_mean, t_mean, adens_mean)
    # ax.set_title('adens_mean')
    # ax.set_xlabel('r')
    # ax.set_ylabel('t')
    # plt.show()
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(r_mean, t_mean, sdens_mean)
    # ax.set_title('sdens_mean')
    # ax.set_xlabel('r')
    # ax.set_ylabel('t')
    # plt.show()
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(r_mean, t_mean, ai_mean)
    # ax.set_title('ai_mean')
    # ax.set_xlabel('r')
    # ax.set_ylabel('t')
    # plt.show()
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(r_mean, t_mean, mf_mean)
    # ax.set_title('mf_mean')
    # ax.set_xlabel('r')
    # ax.set_ylabel('t')
    # plt.show()

    # fifi(15)

    # set_name = 'surface_test'
    # wls,r,t = get_default_P_leaf()
    from src.utils import spectra_utils as SU
    # SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
    # o = Optimization(set_name)
    # o.run_optimization(resolution=20)

    # fig, ax = plt.subplots()
    # P._plot_refl_tran_to_axis(ax, r, t, l, x_label='wavelength', invert_tran=True)
    # plt.show()

    # # Test the software with hard coded data.
    # presets.optimize_default_target(spectral_resolution=50)
    #
    # # Example using "real" data
    # data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    # set_name = 'test_set'
    # o = Optimization(set_name)
    # TH.write_target(set_name, data, sample_id=0)
    # o.run_optimization()

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
    return a * (r**b) * (t**c)


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    from scipy.optimize import curve_fit

    import prospect_d as PD
    import plotter as P
    import matplotlib.pyplot as plt
    from src.data import file_handling as FH
    from src.data import toml_handling as TH
    from src import constants as C

    set_name = 'specchio_5nm'
    # FH.expand(set_name)
    # result_dict = TH.read_set_result(set_name)
    # ids = FH.list_finished_sample_ids(set_name)
    # ad = []
    # sd = []
    # ai = []
    # mf = []
    # r_m = []
    # t_m = []
    # for _, sample_id in enumerate(ids):
    #     result = TH.read_sample_result(set_name, sample_id)
    #     ad.append(result[C.key_sample_result_ad])
    #     sd.append(result[C.key_sample_result_sd])
    #     ai.append(result[C.key_sample_result_ai])
    #     mf.append(result[C.key_sample_result_mf])
    #     r_m.append(result[C.key_sample_result_rm])
    #     t_m.append(result[C.key_sample_result_tm])

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

    set_name = 'surface_test'
    wls,r,t = get_default_P_leaf()
    from src.utils import spectra_utils as SU
    SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
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

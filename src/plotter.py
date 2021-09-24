"""
This file contains plotting-related code.
"""

import os
import logging

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

from src import constants as C
from src import toml_handlling as T
from src import file_handling as FH

figsize = (12,8)
fig_title_font_size = 18

variable_space_ylim = [-0.5, 1]

refl_point_color = 'blue'
tran_point_color = 'orange'

def _plot_list_variable_to_axis(axis_object, label: str, data, skip_first=False):
    """Plots given Blender parameter to given matplolib.axis object.

    :param axis_object:
        matplotlib.axis object to plot to.
    :param label:
        Label for the plot.
    :param data:
        List of parameter values per wavelength.
    :param skip_first:
        If true, the first value is not plotted. This exists because the history given by
        the optimization class contains the starting guess as the first datapoint.
    :return:
        None
    """

    length = len(data)
    if skip_first:
        axis_object.plot(np.arange(length - 1), data[1:length], label=label)
    else:
        axis_object.plot(np.arange(length), data, label=label)


def _plot_x_line_to_axis(axis_object, label: str, data: float, x_values, invert=False):
    """Plots a horizontal line to given axis object on height data. Used for subresult plots.

    :param axis_object:
        matplotlib.axis object to plot to.
    :param label:
        Label for the plot.
    :param data:
        Into what height the horizontal line should be drawn to.
    :param x_values:
        Essentially the length of the line.
    :param invert:
        Use reciprocal of data as height. Used for transmittance values.
    :return:
        None
    """

    if invert:
        axis_object.plot(x_values, 1 - np.ones((len(x_values))) * data, label=label, color='red')
    else:
        axis_object.plot(x_values, np.ones((len(x_values))) * data, label=label, color='red')


def plot_refl_tran_to_axis(axis_object, refl, tran, x_values, x_label, invert_tran=False, skip_first=False,
                           refl_color='blue', tran_color='orange', refl_errors=None, tran_errors=None):
    """Plots reflectance and transmittance to given axis object.

    :param axis_object:
        matplotlib.axis object to plot to.
    :param refl:
        List of reflectance values to be plotted.
    :param tran:
        List of transmittance values to be plotted.
    :param x_values:
        Essentially a list of wavelengths.
    :param x_label:
        Label of x-axis.
    :param invert_tran:
        If True, transmittance is plotted on separate y-axis 'upside down' as is common.
    :param skip_first:
        If true, the first value is not plotted. This exists because the history given by
        the optimization class contains the starting guess as the first datapoint.
    :param refl_color:
        Color of reflectance points.
    :param tran_color:
        Color of transmittance points.
    :return:
        None
    """
    use_errors = False
    if refl_errors is not None and tran_errors is not None:
        use_errors = True
    axis_object.set_xlabel(x_label)
    axis_object.set_ylabel('Reflectance', color=refl_color)
    axis_object.tick_params(axis='y', labelcolor=refl_color)
    # Make twin axis for transmittance
    axt = axis_object.twinx()
    axt.set_ylabel('Transmittance', color=tran_color)
    axt.tick_params(axis='y', labelcolor=tran_color)
    # But use given x_values for plotting
    length = len(x_values)
    marker = '.'
    if skip_first:
        axis_object.scatter(x_values[1:length], refl[1:length], label="Reflectance", color=refl_color, marker=marker)
        axt.scatter(x_values[1:length], tran[1:length], label="Transmittance", color=tran_color, marker=marker)
    else:
        if use_errors:
            axis_object.errorbar(x_values, refl, yerr=refl_errors, ls='', label="Reflectance", color=refl_color, marker=marker)
            axt.errorbar(x_values, tran, yerr=tran_errors, ls='', label="Transmittance", color=tran_color, marker=marker)
        else:
            axis_object.scatter(x_values, refl, label="Reflectance", color=refl_color, marker=marker)
            axt.scatter(x_values, tran, label="Transmittance", color=tran_color, marker=marker)

    axis_object.set_ylim([0, 1])
    if invert_tran:
        axt.set_ylim([1, 0])
    else:
        axt.set_ylim([0, 1])


def plot_subresult_opt_history(set_name: str, wl: float, sample_id, dont_show=True, save_thumbnail=True):
    """Plots otimization history of a single wavelength using existing subresult toml file.

    :param sample_id:
    :param set_name:
        Set name.
    :param wl:
        Wavelength of the optimization.
    :param save_thumbnail:
        If True, a JPG image is saved to result/plot folder. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    :return:
        None
    """

    subres_dict = T.read_subresult(set_name=set_name, wl=wl, sample_id=sample_id)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Optimization history (wl: {wl:.2f} nm)", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_absorption_density,
                                subres_dict[C.subres_key_history_absorption_density], skip_first=True)
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_scattering_density,
                                subres_dict[C.subres_key_history_scattering_density], skip_first=True)
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_scattering_anisotropy,
                                subres_dict[C.subres_key_history_scattering_anisotropy], skip_first=True)
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_mix_factor,
                                subres_dict[C.subres_key_history_mix_factor], skip_first=True)
    ax[0].set_xlabel('Render call')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    _plot_x_line_to_axis(ax[1], C.subres_key_reflectance_measured, subres_dict[C.subres_key_reflectance_measured],
                         np.arange(1,len(subres_dict[C.subres_key_history_reflectance])))
    _plot_x_line_to_axis(ax[1], C.subres_key_transmittance_measured, subres_dict[C.subres_key_transmittance_measured],
                         np.arange(1,len(subres_dict[C.subres_key_history_transmittance])), invert=True)
    plot_refl_tran_to_axis(ax[1], subres_dict[C.subres_key_history_reflectance],
                           subres_dict[C.subres_key_history_transmittance],
                           np.arange(len(subres_dict[C.subres_key_history_scattering_anisotropy])),
                           'Render call', invert_tran=True,
                           skip_first=True)

    if save_thumbnail is not None:
        folder = FH.get_path_opt_subresult(set_name, sample_id)
        image_name = f"subresplot_wl{wl:.2f}.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the subresult plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)

def plot_averaged_sample_result(set_name: str, dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Averaged optimization result", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    marker = '.'

    ids = FH.list_finished_sample_ids(set_name)
    wls = []
    adens = []
    sdens = []
    ai = []
    mf = []
    r_m = []
    t_m = []
    r = []
    t = []
    for _, sample_id in enumerate(ids):
        result = T.read_sample_result(set_name, sample_id)
        wls = result[C.result_key_wls]
        adens.append(result[C.result_key_absorption_density])
        sdens.append(result[C.result_key_scattering_density])
        ai.append(result[C.result_key_scattering_anisotropy])
        mf.append(result[C.result_key_mix_factor])
        r_m.append(result[C.result_key_refls_measured])
        t_m.append(result[C.result_key_trans_measured])
        r.append(result[C.result_key_refls_modeled])
        t.append(result[C.result_key_trans_modeled])

    adens_mean = np.array(adens).mean(axis=0)
    sdens_mean = np.array(sdens).mean(axis=0)
    ai_mean = np.array(ai).mean(axis=0)
    mf_mean = np.array(mf).mean(axis=0)
    r_m_mean = np.array(r_m).mean(axis=0)
    t_m_mean = np.array(t_m).mean(axis=0)
    r_mean = np.array(r).mean(axis=0)
    t_mean = np.array(t).mean(axis=0)
    # half of the std as plt.errorbar plots the same to top and bottom
    adens_std = np.array(adens).std(axis=0) / 2
    sdens_std = np.array(sdens).std(axis=0) / 2
    ai_std = np.array(ai).std(axis=0) / 2
    mf_std = np.array(mf).std(axis=0) / 2
    r_m_std= np.array(r_m).std(axis=0) / 2
    t_m_std= np.array(t_m).std(axis=0) / 2
    r_std = np.array(r).std(axis=0) / 2
    t_std = np.array(t).std(axis=0) / 2

    ax[0].errorbar(wls, adens_mean, ls='', yerr=adens_std, label=C.result_key_absorption_density, marker=marker)
    ax[0].errorbar(wls, sdens_mean, ls='', yerr=sdens_std, label=C.result_key_scattering_density, marker=marker)
    ax[0].errorbar(wls, ai_mean,    ls='', yerr=ai_std, label=C.result_key_scattering_anisotropy, marker=marker)
    ax[0].errorbar(wls, mf_mean,    ls='', yerr=mf_std, label=C.result_key_mix_factor, marker=marker)
    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label)
    # ax[1].set_xlabel('Wavelength')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    plot_refl_tran_to_axis(ax[1], r_m_mean, t_m_mean, result[C.result_key_wls], x_label, invert_tran=True,
                           tran_color='black', refl_color='black', skip_first=False, refl_errors=r_m_std,
                           tran_errors=t_m_std)
    plot_refl_tran_to_axis(ax[1], r_mean, t_mean, result[C.result_key_wls], x_label, invert_tran=True, skip_first=False,
                           refl_errors=r_std, tran_errors=t_std)
    if save_thumbnail:
        folder = FH.get_set_result_folder_path(set_name)
        image_name = f"set_average_result_plot.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the result plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

def plot_averaged_sample_errors(set_name: str, dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.suptitle(f"Optimization errors ", fontsize=fig_title_font_size)
    # ax.set_title('Optimization errors')
    marker = '.'

    ids = FH.list_finished_sample_ids(set_name)
    wls = []
    refl_errs = []
    tran_errs = []
    for _,sample_id in enumerate(ids):
        result = T.read_sample_result(set_name, sample_id)
        wls = result[C.result_key_wls]
        refl_errs.append(result[C.result_key_refls_error])
        tran_errs.append(result[C.result_key_trans_error])

    refl_errs_mean = np.array(refl_errs).mean(axis=0)
    tran_errs_mean = np.array(tran_errs).mean(axis=0)
    refl_errs_std = np.array(refl_errs).std(axis=0) / 2
    tran_errs_std = np.array(tran_errs).std(axis=0) / 2

    # x_data = result[C.result_key_wls]
    ax.errorbar(wls, refl_errs_mean, yerr=refl_errs_std, alpha=1.0, ls='', label=C.result_key_refls_error + ' mean', marker=marker, color=refl_point_color)
    ax.errorbar(wls, tran_errs_mean, yerr=tran_errs_std, alpha=1.0, ls='', label=C.result_key_trans_error + ' mean', marker=marker, color=tran_point_color)
    x_label = 'Wavelength [nm]'
    ax.set_xlabel(x_label)
    ax.legend()
    # ax.set_ylim(variable_space_ylim)

    if save_thumbnail:
        folder = FH.get_set_result_folder_path(set_name)
        image_name = f"set_error_plot.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the result plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

def plot_sample_result(set_name: str, sample_id, dont_show=True, save_thumbnail=True):
    """Plots final result of all optimized wavelengths to result/plot folder using existing final result TOML file.

    :param sample_id:
    :param set_name:
        Set name.
    :param save_thumbnail:
        If True, a JPG image is saved to result/plot folder. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    :return:
        None
    """

    result = T.read_sample_result(set_name, sample_id)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Optimization result ", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    x_data = result[C.result_key_wls]
    marker = '.'
    ax[0].scatter(x_data, result[C.result_key_absorption_density], label=C.result_key_absorption_density, marker=marker)
    ax[0].scatter(x_data, result[C.result_key_scattering_density], label=C.result_key_scattering_density, marker=marker)
    ax[0].scatter(x_data, result[C.result_key_scattering_anisotropy], label=C.result_key_scattering_anisotropy, marker=marker)
    ax[0].scatter(x_data, result[C.result_key_mix_factor], label=C.result_key_mix_factor, marker=marker)
    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label)
    # ax[1].set_xlabel('Wavelength')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    plot_refl_tran_to_axis(ax[1], result[C.result_key_refls_measured], result[C.result_key_trans_measured],
                           result[C.result_key_wls], x_label, invert_tran=True, tran_color='black',
                           refl_color='black', skip_first=False)
    plot_refl_tran_to_axis(ax[1], result[C.result_key_refls_modeled], result[C.result_key_trans_modeled],
                           result[C.result_key_wls], x_label, invert_tran=True, skip_first=False)
    if save_thumbnail:
        folder = FH.get_set_result_folder_path(set_name)
        image_name = f"sample_{sample_id}_result_plot.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the result plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()


def plot_vars_per_absorption(result_dict, degree=2):
    """Prints polynomial fitting coefficients.

    Used to get the coefficients for starting guess. This should be run for the result of optimizing
    spectra_utils.make_linear_test_target().

    TODO automize the whole thing and save coefficients to a file.

    :param result_dict:
        Result dict to fit the polynomials to. As returned by toml_handling.read_final_result(set_name).
    :param degree:
        Degree of the polynomial to be fit. Default is 2.
    :return:
        Coefficients in a list starting from the highest order, e.g., [A, B, C] in Ax^2 + Bx + C.
    """

    # print(result_dict)
    def fit_poly(x,y,degree,name):
        fit = Polynomial.fit(x, y, deg=degree, domain=[0, 1])
        coeffs = fit.convert().coef
        print(f"fitting coeffs for {name}: {coeffs}")
        y = np.array([np.sum(np.array([coeffs[i] * (j ** i) for i in range(len(coeffs))])) for j in x])
        plt.plot(x, y, color='black')
        return coeffs

    wls = result_dict[C.result_key_wls]
    r_list = np.array([r for _, r in sorted(zip(wls, result_dict[C.result_key_refls_modeled]))])
    t_list = np.array([t for _, t in sorted(zip(wls, result_dict[C.result_key_trans_modeled]))])
    ad_list = np.array([ad for _, ad in sorted(zip(wls, result_dict[C.result_key_absorption_density]))])
    sd_list = np.array([sd for _, sd in sorted(zip(wls, result_dict[C.result_key_scattering_density]))])
    ai_list = np.array([ai for _, ai in sorted(zip(wls, result_dict[C.result_key_scattering_anisotropy]))])
    mf_list = np.array([mf for _, mf in sorted(zip(wls, result_dict[C.result_key_mix_factor]))])
    a_list = np.ones_like(r_list) - (r_list + t_list) # modeled absorptions
    fit_poly(a_list,ad_list,degree=degree, name=C.result_key_absorption_density)
    fit_poly(a_list,sd_list,degree=degree, name=C.result_key_scattering_density)
    fit_poly(a_list,ai_list,degree=degree, name=C.result_key_scattering_anisotropy)
    fit_poly(a_list,mf_list,degree=degree, name=C.result_key_mix_factor)
    plt.scatter(a_list, ad_list, label=C.result_key_absorption_density)
    plt.scatter(a_list, sd_list, label=C.result_key_scattering_density)
    plt.scatter(a_list, ai_list, label=C.result_key_scattering_anisotropy)
    plt.scatter(a_list, mf_list, label=C.result_key_mix_factor)
    plt.xlabel('Absorption')
    plt.legend()
    plt.show()
    # print(a_list)

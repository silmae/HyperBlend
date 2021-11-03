"""
This file contains plotting-related code.

Tips for plotting:
https://towardsdatascience.com/5-powerful-tricks-to-visualize-your-data-with-matplotlib-16bc33747e05

"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from src import constants as C
from src.data import file_handling as FH, toml_handling as T

figsize = (12,6)
figsize_single = (6,6)
fig_title_font_size = 18
axis_label_font_size = 16

variable_space_ylim = [0.0, 1]

color_reflectance = 'royalblue'
color_transmittance = 'deeppink'
color_reflectance_measured = 'black'
color_transmittance_measured = 'black'
color_ad = 'olivedrab'
color_sd = 'darkorange'
color_ai = 'brown'
color_mf = 'darkorchid'
alpha_error = 0.2
max_ticks = 8


def _plot_list_variable_to_axis(axis_object, label: str, data, color, skip_first=False):
    """Plots given Blender parameter to given matplolib.axis object.

    :param color:
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
        axis_object.plot(np.arange(length - 1), data[1:length], label=label, color=color)
    else:
        axis_object.plot(np.arange(length), data, label=label, color=color)


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


def plot_refl_tran_to_axis(axis_object, refl, tran, x_values, x_label, invert_tran=False, skip_first=False, refl_color=color_reflectance, tran_color=color_transmittance):
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
        axis_object.plot(x_values[1:length], refl[1:length], label="Reflectance", color=refl_color, marker=marker)
        axt.plot(x_values[1:length], tran[1:length], label="Transmittance", color=tran_color, marker=marker)
    else:
        axis_object.plot(x_values, refl, label="Reflectance", color=refl_color, marker=marker)
        axt.plot(x_values, tran, label="Transmittance", color=tran_color, marker=marker)

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
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_absorption_density, subres_dict[C.subres_key_history_absorption_density], color=color_ad, skip_first=True)
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_scattering_density, subres_dict[C.subres_key_history_scattering_density], color=color_sd, skip_first=True)
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_scattering_anisotropy, subres_dict[C.subres_key_history_scattering_anisotropy], color=color_ai, skip_first=True)
    _plot_list_variable_to_axis(ax[0], C.subres_key_history_mix_factor, subres_dict[C.subres_key_history_mix_factor], color=color_mf, skip_first=True)
    ax[0].set_xlabel('Render call')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    _plot_x_line_to_axis(ax[1], C.subres_key_reflectance_measured, subres_dict[C.subres_key_reflectance_measured],
                         np.arange(1,len(subres_dict[C.subres_key_history_reflectance])))
    _plot_x_line_to_axis(ax[1], C.subres_key_transmittance_measured, subres_dict[C.subres_key_transmittance_measured],
                         np.arange(1,len(subres_dict[C.subres_key_history_transmittance])), invert=True)
    plot_refl_tran_to_axis(ax[1], subres_dict[C.subres_key_history_reflectance], subres_dict[C.subres_key_history_transmittance], np.arange(len(subres_dict[C.subres_key_history_scattering_anisotropy])), 'Render call', invert_tran=True, skip_first=True)

    if save_thumbnail is not None:
        folder = FH.path_directory_subresult(set_name, sample_id)
        image_name = f"subresplot_wl{wl:.2f}.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the subresult plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)

def plot_neat_errors(ax_obj, x, value, value_std, color, label, ls='-'):
    sorting_idx = x.argsort()
    x_sorted = x[sorting_idx[::-1]]
    value_sorted = value[sorting_idx[::-1]]
    std_sorted = value_std[sorting_idx[::-1]]
    ax_obj.fill_between(x_sorted, value_sorted-std_sorted, value_sorted+std_sorted, alpha=alpha_error, color=color)
    ax_obj.plot(x_sorted, value_sorted, color=color, ls=ls, label=label)

def plot_averaged_sample_result(set_name: str, dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # fig.suptitle(f"Averaged optimization result", fontsize=fig_title_font_size)
    # ax[0].set_title('Variable space')
    # ax[1].set_title('Target space')
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

    wls = np.array(wls)
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

    plot_neat_errors(ax[0], wls, adens_mean, adens_std, color_ad, 'Absorption density')
    plot_neat_errors(ax[0], wls, sdens_mean, sdens_std, color_sd, 'Scattering density')
    plot_neat_errors(ax[0], wls, ai_mean, ai_std, color_ai, 'Scattering anistropy')
    plot_neat_errors(ax[0], wls, mf_mean, mf_std, color_mf, 'Mix factor')

    # ax[
    #     0].errorbar(wls, sdens_mean, errorevery=error_every, ls='', yerr=sdens_std, label=C.result_key_scattering_density, marker=marker)
    # ax[
    #     0].errorbar(wls, ai_mean, errorevery=error_every, ls='', yerr=ai_std, label=C.result_key_scattering_anisotropy, marker=marker)
    # ax[
    #     0].errorbar(wls, mf_mean, errorevery=error_every, ls='', yerr=mf_std, label=C.result_key_mix_factor, marker=marker)

    # error_every = 5
    # ax[0].errorbar(wls, adens_mean, errorevery=error_every,ls='', yerr=adens_std, label=C.result_key_absorption_density, marker=marker)
    # ax[0].errorbar(wls, sdens_mean, errorevery=error_every,ls='', yerr=sdens_std, label=C.result_key_scattering_density, marker=marker)
    # ax[0].errorbar(wls, ai_mean,    errorevery=error_every,ls='', yerr=ai_std, label=C.result_key_scattering_anisotropy, marker=marker)
    # ax[0].errorbar(wls, mf_mean,    errorevery=error_every,ls='', yerr=mf_std, label=C.result_key_mix_factor, marker=marker)
    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=axis_label_font_size)
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    # ax[1].set_xlabel('Wavelength')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    ax[0].set_ylabel('Material parameter', fontsize=axis_label_font_size)

    ax[1].set_ylim([0,1])
    ax[1].set_ylabel('Reflectance', color=color_reflectance, fontsize=axis_label_font_size)
    ax[1].tick_params(axis='y', labelcolor=color_reflectance)
    plot_neat_errors(ax[1], wls, r_mean, r_std, color_reflectance, 'Reflectance')
    # plot_neat_errors(ax[1], wls, r_m_mean, r_m_std, color_reflectance_measured, 'Reflectance measured', ls='dotted')
    ax[1].plot(wls, r_m_mean, color=color_reflectance_measured, ls='dotted')

    ax_inverted = ax[1].twinx()
    ax_inverted.set_ylim([1, 0])
    ax_inverted.set_ylabel('Transmittance', color=color_transmittance, fontsize=axis_label_font_size)
    ax_inverted.tick_params(axis='y', labelcolor=color_transmittance)
    plot_neat_errors(ax_inverted, wls, t_mean, t_std, color_transmittance, 'Transmittance')
    # plot_neat_errors(ax_inverted, wls, t_m_mean, t_m_std, color_transmittance_measured, 'Transmittance measured', ls='dotted')
    ax_inverted.plot(wls, t_m_mean, color=color_transmittance_measured, ls='dotted')

    # plot_refl_tran_to_axis(ax[1], r_m_mean, t_m_mean, result[C.result_key_wls], x_label, invert_tran=True,
    #                        tran_color='black', refl_color='black', skip_first=False, refl_errors=r_m_std,
    #                        tran_errors=t_m_std)
    # plot_refl_tran_to_axis(ax[1], r_mean, t_mean, result[C.result_key_wls], x_label, invert_tran=True, skip_first=False,
    #                        refl_errors=r_std, tran_errors=t_std)
    if save_thumbnail:
        folder = FH.path_directory_set_result(set_name)
        image_name = f"set_average_result_plot.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the result plot to '{path}'.")
        plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()

def plot_averaged_sample_errors(set_name: str, dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Optimization errors ", fontsize=fig_title_font_size)
    # ax.set_title('Optimization errors')
    marker = '.'

    ax.set_ylabel('RMSE', fontsize=axis_label_font_size)

    ids = FH.list_finished_sample_ids(set_name)
    wls = []
    refl_errs = []
    tran_errs = []
    for _,sample_id in enumerate(ids):
        result = T.read_sample_result(set_name, sample_id)
        wls = result[C.result_key_wls]
        refl_errs.append(result[C.result_key_refls_error])
        tran_errs.append(result[C.result_key_trans_error])

    wls = np.array(wls)
    refl_errs_mean = np.array(refl_errs).mean(axis=0)
    tran_errs_mean = np.array(tran_errs).mean(axis=0)
    refl_errs_std = np.array(refl_errs).std(axis=0) / 2
    tran_errs_std = np.array(tran_errs).std(axis=0) / 2

    # x_data = result[C.result_key_wls]
    # plot_neat_errors(ax, wls, refl_errs_mean, refl_errs_std, color_reflectance, 'Reflectance error')
    # plot_neat_errors(ax, wls, tran_errs_mean, tran_errs_std, color_transmittance, 'Transmittance error')
    # ax.scatter(wls, refl_errs_mean, color=color_reflectance,   marker_size=2)
    # ax.scatter(wls, tran_errs_mean, color=color_transmittance, marker_size=2)
    error_every = 5
    ax.errorbar(wls, refl_errs_mean, yerr=refl_errs_std, errorevery=error_every, alpha=1.0, ls='', lw=0., label='Reflectance error',   marker='x', markersize=4, color=color_reflectance)
    ax.errorbar(wls, tran_errs_mean, yerr=tran_errs_std, errorevery=error_every, alpha=1.0, ls='', lw=0., label='Transmittance error', marker=marker, markersize=4, color=color_transmittance)
    x_label = 'Wavelength [nm]'
    ax.set_xlabel(x_label, fontsize=axis_label_font_size)
    ax.legend()
    # ax.set_ylim(variable_space_ylim)

    if save_thumbnail:
        folder = FH.path_directory_set_result(set_name)
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
    plot_refl_tran_to_axis(ax[1], result[C.result_key_refls_measured], result[C.result_key_trans_measured], result[C.result_key_wls], x_label, invert_tran=True, skip_first=False, refl_color='black', tran_color='black')
    plot_refl_tran_to_axis(ax[1], result[C.result_key_refls_modeled], result[C.result_key_trans_modeled], result[C.result_key_wls], x_label, invert_tran=True, skip_first=False)
    if save_thumbnail:
        folder = FH.path_directory_set_result(set_name)
        image_name = f"sample_{sample_id}_result_plot.png"
        path = os.path.normpath(folder + '/' + image_name)
        logging.info(f"Saving the result plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()


def plot_vars_per_absorption(dont_show=True, save_thumbnail=True):

    set_name = C.starting_guess_set_name
    result_dict = T.read_sample_result(set_name, 0)
    coeffs = T.read_starting_guess_coeffs()
    wls = result_dict[C.result_key_wls]
    r_list = np.array([r for _, r in sorted(zip(wls, result_dict[C.result_key_refls_modeled]))])
    t_list = np.array([t for _, t in sorted(zip(wls, result_dict[C.result_key_trans_modeled]))])
    ad_list = np.array([ad for _, ad in sorted(zip(wls, result_dict[C.result_key_absorption_density]))])
    sd_list = np.array([sd for _, sd in sorted(zip(wls, result_dict[C.result_key_scattering_density]))])
    ai_list = np.array([ai for _, ai in sorted(zip(wls, result_dict[C.result_key_scattering_anisotropy]))])
    mf_list = np.array([mf for _, mf in sorted(zip(wls, result_dict[C.result_key_mix_factor]))])
    a_list = np.ones_like(r_list) - (r_list + t_list) # modeled absorptions
    plt.scatter(a_list, ad_list, label=C.result_key_absorption_density)
    plt.scatter(a_list, sd_list, label=C.result_key_scattering_density)
    plt.scatter(a_list, ai_list, label=C.result_key_scattering_anisotropy)
    plt.scatter(a_list, mf_list, label=C.result_key_mix_factor)
    for _,key in enumerate(coeffs):
        coeff  = coeffs[key]
        y = np.array([np.sum(np.array([coeff[i] * (j ** i) for i in range(len(coeff))])) for j in a_list])
        plt.plot(a_list, y, color='black')

    plt.xlabel('Absorption', fonsize=axis_label_font_size)
    plt.legend()

    if save_thumbnail:
        p = FH.path_directory_set_result(set_name)
        image_name = f"variable_fitting.png"
        path = os.path.normpath(p + '/' + image_name)
        logging.info(f"Saving variable fitting plot to '{path}'.")
        plt.savefig(path, dpi=300)

    if not dont_show:
        plt.show()

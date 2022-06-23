"""
This file contains plotting-related code.

Tips for plotting:
https://towardsdatascience.com/5-powerful-tricks-to-visualize-your-data-with-matplotlib-16bc33747e05

"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src import constants as C
from src.data import file_handling as FH, toml_handling as T, file_names as FN, path_handling as P


figsize = (12,6)
"""Figure size for two plot figures."""
figsize_single = (6,6)
"""Figure size for single plot figures."""
fig_title_font_size = 18
"""Title font size."""
axis_label_font_size = 16
"""Axis label font size"""

variable_space_ylim = [0.0, 1.0]
"""Y-axis limit for leaf material parameter plot."""

# Colors
color_reflectance = 'royalblue'
color_transmittance = 'deeppink'
color_reflectance_measured = 'black'
color_transmittance_measured = 'black'
color_ad = 'olivedrab'
color_sd = 'darkorange'
color_ai = 'brown'
color_mf = 'darkorchid'
color_history_target = 'black'

alpha_error = 0.2
"""Alpha for std shadow."""

max_ticks = 8
"""Max tick count for wavelength."""

image_type = 'png'


def plot_3d_rt(r,t,z, z_label,z_intensity=None,surface_parameters=None,fittable=None,save_thumbnail=True,show_plot=False):
    # setup figure object
    fig = plt.figure(figsize=figsize_single)
    ax = plt.axes(projection="3d")
    ax.set_xlabel('R')
    ax.set_ylabel('T')
    ax.set_zlabel(z_label)
    ax.elev = 30
    ax.azim = 225
    num_points = 25
    R, T = np.meshgrid(np.linspace(0, max(r), num_points), np.linspace(0, max(t), num_points))

    if z_intensity is None:
        z_intens = z
    else:
        z_intens = z_intensity
    ax.scatter(r, t, z, c=z_intens, cmap=plt.cm.hot)

    if surface_parameters is not None:
        Z = fittable(np.array([R, T]), *surface_parameters)
        ax.plot_surface(R, T, Z, alpha=0.5)

    if save_thumbnail:
        folder = P.path_directory_surface_model()
        image_name = f"{z_label}.png"
        path = P.join(folder, image_name)
        logging.info(f"Saving surface plot to '{path}'.")
        plt.tight_layout()
        plt.savefig(path, dpi=300)

    plt.show()


def plot_nn_train_history(train_loss, test_loss, best_epoch_idx, dont_show=True, save_thumbnail=True) -> None:
    """Plots optimization history of a single wavelength using existing wavelength result toml file.

    :param train_loss:
        Training loss
    :param test_loss:
        Validation loss
    :param best_epoch_idx:
        Index of the best epoch
    :param save_thumbnail:
        If True, a PNG image is saved to result/plot folder. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Training history", fontsize=fig_title_font_size)
    ax.plot(train_loss, label="Train loss")
    ax.plot(test_loss, label="Test loss")
    ax.scatter(best_epoch_idx, test_loss[best_epoch_idx], facecolors='none', edgecolors='r')
    ax.set_xlabel('Epoch')
    ax.legend()

    if save_thumbnail:
        folder = P.path_directory_surface_model()
        image_name = "nn_train_history.png"
        path = P.join(folder, image_name)
        logging.info(f"Saving NN training history to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)


def plot_wl_optimization_history(set_name: str, wl: float, sample_id, dont_show=True, save_thumbnail=True) -> None:
    """Plots optimization history of a single wavelength using existing wavelength result toml file.

    :param set_name:
        Set name.
    :param wl:
        Wavelength of the optimization.
    :param sample_id:
        Sample id.
    :param save_thumbnail:
        If True, a PNG image is saved to result/plot folder. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    """

    subres_dict = T.read_wavelength_result(set_name=set_name, wl=wl, sample_id=sample_id)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Optimization history (wl: {wl:.2f} nm)", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_ad])), subres_dict[C.key_wl_result_history_ad], label=C.key_wl_result_history_ad, color=color_ad)
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_sd])), subres_dict[C.key_wl_result_history_sd], label=C.key_wl_result_history_sd, color=color_sd)
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_ai])), subres_dict[C.key_wl_result_history_ai], label=C.key_wl_result_history_ai, color=color_ai)
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_mf])), subres_dict[C.key_wl_result_history_mf], label=C.key_wl_result_history_mf, color=color_mf)
    ax[0].set_xlabel('Render call')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)

    # Plot horizontal line to location of measured value
    x_data = np.arange(1, len(subres_dict[C.key_wl_result_history_r]))
    ax[1].plot(x_data, np.ones(len(x_data)) * subres_dict[C.key_wl_result_refl_measured], label=C.key_wl_result_refl_measured, color=color_history_target)
    ax[1].plot(x_data, 1 - np.ones(len(x_data)) * subres_dict[C.key_wl_result_tran_measured], label=C.key_wl_result_tran_measured, color=color_history_target)

    _plot_refl_tran_to_axis(ax[1], subres_dict[C.key_wl_result_history_r], subres_dict[C.key_wl_result_history_t], np.arange(len(subres_dict[C.key_wl_result_history_ai])), 'Render call', invert_tran=True)

    if save_thumbnail is not None:
        folder = P.path_directory_subresult(set_name, sample_id)
        image_name = FN.filename_wl_result_plot(wl)
        path = P.join(folder, image_name)
        logging.info(f"Saving the subresult plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)


def plot_set_result(set_name: str, dont_show=True, save_thumbnail=True) -> None:
    """Plot average of sample results as the set result.

    :param set_name:
        Set name.
    :param dont_show:
        If False, pyplot.show() is called, otherwise nothing is shown. Default is True.
    :param save_thumbnail:
        If True, save plot to disk. Default is True.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # fig.suptitle(f"Averaged optimization result", fontsize=fig_title_font_size)
    # ax[0].set_title('Variable space')
    # ax[1].set_title('Target space')

    r = T.read_set_result(set_name)
    wls = r[C.key_set_result_wls]
    ad_mean = np.array(r[C.key_set_result_wl_ad_mean])
    sd_mean = np.array(r[C.key_set_result_wl_sd_mean])
    ai_mean = np.array(r[C.key_set_result_wl_ai_mean])
    mf_mean = np.array(r[C.key_set_result_wl_mf_mean])
    r_mean  = np.array(r[C.key_set_result_wl_r_mean])
    t_mean  = np.array(r[C.key_set_result_wl_t_mean])
    rm_mean = np.array(r[C.key_set_result_wl_rm_mean])
    tm_mean = np.array(r[C.key_set_result_wl_tm_mean])
    ad_std  = np.array(r[C.key_set_result_wl_ad_std])
    sd_std  = np.array(r[C.key_set_result_wl_sd_std])
    ai_std  = np.array(r[C.key_set_result_wl_ai_std])
    mf_std  = np.array(r[C.key_set_result_wl_mf_std])
    r_std   = np.array(r[C.key_set_result_wl_r_std])
    t_std   = np.array(r[C.key_set_result_wl_t_std])
    rm_std  = np.array(r[C.key_set_result_wl_rm_std])
    tm_std  = np.array(r[C.key_set_result_wl_tm_std])

    _plot_with_shadow(ax[0], wls, ad_mean, ad_std, color_ad, 'Absorption density')
    _plot_with_shadow(ax[0], wls, sd_mean, sd_std, color_sd, 'Scattering density')
    _plot_with_shadow(ax[0], wls, ai_mean, ai_std, color_ai, 'Scattering anistropy')
    _plot_with_shadow(ax[0], wls, mf_mean, mf_std, color_mf, 'Mix factor')

    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=axis_label_font_size)
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    ax[0].set_ylabel('Material parameter', fontsize=axis_label_font_size)

    ax[1].set_ylim([0,1])
    ax[1].set_ylabel('Reflectance', color=color_reflectance, fontsize=axis_label_font_size)
    ax[1].tick_params(axis='y', labelcolor=color_reflectance)
    _plot_with_shadow(ax[1], wls, r_mean, r_std, color_reflectance, 'Reflectance')
    ax[1].plot(wls, rm_mean, color=color_reflectance_measured, ls='dotted')
    ax[1].plot(wls, rm_mean - (rm_std/2), color='gray', ls='dashed')
    ax[1].plot(wls, rm_mean + (rm_std/2), color='gray', ls='dashed')

    ax_inverted = ax[1].twinx()
    ax_inverted.set_ylim([1, 0])
    ax_inverted.set_ylabel('Transmittance', color=color_transmittance, fontsize=axis_label_font_size)
    ax_inverted.tick_params(axis='y', labelcolor=color_transmittance)
    _plot_with_shadow(ax_inverted, wls, t_mean, t_std, color_transmittance, 'Transmittance')
    ax_inverted.plot(wls, tm_mean, color=color_transmittance_measured, ls='dotted')
    ax_inverted.plot(wls, tm_mean - (tm_std / 2), color='gray', ls='dashed')
    ax_inverted.plot(wls, tm_mean + (tm_std / 2), color='gray', ls='dashed')

    if save_thumbnail:
        folder = P.path_directory_set_result(set_name)
        image_name = FN.filename_set_result_plot()
        path = P.join(folder, image_name)
        logging.info(f"Saving the set result plot to '{path}'.")
        plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()


def plot_set_errors(set_name: str, dont_show=True, save_thumbnail=True):
    """Plots averaged optimization errors of a sample. """

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
        wls = result[C.key_sample_result_wls]
        refl_errs.append(result[C.key_sample_result_re])
        tran_errs.append(result[C.key_sample_result_te])

    wls = np.array(wls)
    refl_errs_mean = np.array(refl_errs).mean(axis=0)
    tran_errs_mean = np.array(tran_errs).mean(axis=0)
    refl_errs_std = np.array(refl_errs).std(axis=0)
    tran_errs_std = np.array(tran_errs).std(axis=0)

    # x_data = result[C.result_key_wls]
    # plot_neat_errors(ax, wls, refl_errs_mean, refl_errs_std, color_reflectance, 'Reflectance error')
    # plot_neat_errors(ax, wls, tran_errs_mean, tran_errs_std, color_transmittance, 'Transmittance error')
    # ax.scatter(wls, refl_errs_mean, color=color_reflectance,   marker_size=2)
    # ax.scatter(wls, tran_errs_mean, color=color_transmittance, marker_size=2)
    error_every = 5
    line_width = 0.0 # does not draw line STD if linewidth is 0.0
    ax.errorbar(wls, refl_errs_mean, yerr=refl_errs_std / 2, errorevery=error_every, alpha=1.0, ls='', lw=line_width, label='Reflectance error',   marker='x', markersize=4, color=color_reflectance)
    ax.errorbar(wls, tran_errs_mean, yerr=tran_errs_std / 2, errorevery=error_every, alpha=1.0, ls='', lw=line_width, label='Transmittance error', marker=marker, markersize=4, color=color_transmittance)
    x_label = 'Wavelength [nm]'

    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax.set_xlabel(x_label, fontsize=axis_label_font_size)
    ax.legend()
    # ax.set_ylim(variable_space_ylim)

    if save_thumbnail:
        folder = P.path_directory_set_result(set_name)
        image_name = FN.filename_set_error_plot()
        path = P.join(folder, image_name)
        logging.info(f"Saving the set error plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()


def plot_sample_result(set_name: str, sample_id: int, dont_show=True, save_thumbnail=True) -> None:
    """Plots sample result.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id.
    :param save_thumbnail:
        If True, a PNG image is saved. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    """

    result = T.read_sample_result(set_name, sample_id)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Optimization result ", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    x_data = result[C.key_sample_result_wls]
    marker = '.'
    ax[0].plot(x_data, result[C.key_sample_result_ad], label=C.key_sample_result_ad, marker=marker, color=color_ad)
    ax[0].plot(x_data, result[C.key_sample_result_sd], label=C.key_sample_result_sd, marker=marker, color=color_sd)
    ax[0].plot(x_data, result[C.key_sample_result_ai], label=C.key_sample_result_ai, marker=marker, color=color_ai)
    ax[0].plot(x_data, result[C.key_sample_result_mf], label=C.key_sample_result_mf, marker=marker, color=color_mf)
    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label)
    # ax[1].set_xlabel('Wavelength')
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    _plot_refl_tran_to_axis(ax[1], result[C.key_sample_result_rm], result[C.key_sample_result_tm], result[C.key_sample_result_wls], x_label, invert_tran=True, refl_color='black', tran_color='black')
    _plot_refl_tran_to_axis(ax[1], result[C.key_sample_result_r], result[C.key_sample_result_t], result[C.key_sample_result_wls], x_label, invert_tran=True)
    if save_thumbnail:
        folder = P.path_directory_set_result(set_name)
        image_name = FN.filename_sample_result_plot(sample_id=sample_id)
        path = P.join(folder, image_name)
        logging.info(f"Saving the sample result plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()


def replot_wl_results(set_name: str):
    """Replot wavelength results.

    Overwrites existing plots.
    """

    sample_ids = FH.list_finished_sample_ids(set_name)
    for sample_id in sample_ids:
        d = T.read_sample_result(set_name, sample_id=sample_id)
        wls = d[C.key_sample_result_wls]
        for wl in wls:
            plot_wl_optimization_history(set_name, wl=wl, sample_id=sample_id)


def _plot_starting_guess_coeffs_fitting(dont_show=True, save_thumbnail=True) -> None:
    """Plot starting guess poynomial fit with data.

    Used only when generating the starting guess.
    """

    set_name = C.starting_guess_set_name
    result_dict = T.read_sample_result(set_name, 0)
    coeffs = T.read_starting_guess_coeffs()
    wls = result_dict[C.key_sample_result_wls]
    r_list = np.array([r for _, r in sorted(zip(wls, result_dict[C.key_sample_result_r]))])
    t_list = np.array([t for _, t in sorted(zip(wls, result_dict[C.key_sample_result_t]))])
    ad_list = np.array([ad for _, ad in sorted(zip(wls, result_dict[C.key_sample_result_ad]))])
    sd_list = np.array([sd for _, sd in sorted(zip(wls, result_dict[C.key_sample_result_sd]))])
    ai_list = np.array([ai for _, ai in sorted(zip(wls, result_dict[C.key_sample_result_ai]))])
    mf_list = np.array([mf for _, mf in sorted(zip(wls, result_dict[C.key_sample_result_mf]))])
    a_list = np.ones_like(r_list) - (r_list + t_list) # modeled absorptions
    ms = 10 # markersize
    ls = 2 # linesize
    plt.scatter(a_list, ad_list, label='Absorption density', color=color_ad, s=ms)
    plt.scatter(a_list, sd_list, label='Scattering density', color=color_sd, s=ms)
    plt.scatter(a_list, ai_list, label='Scattering anisotropy', color=color_ai, s=ms)
    plt.scatter(a_list, mf_list, label='Mix factor', color=color_mf, s=ms)
    for _,key in enumerate(coeffs):
        coeff  = coeffs[key]
        y = np.array([np.sum(np.array([coeff[i] * (j ** i) for i in range(len(coeff))])) for j in a_list])
        plt.plot(a_list, y, color='black', linewidth=ls)

    plt.xlabel('Absorption', fontsize=axis_label_font_size)
    plt.ylabel('Material parameter', fontsize=axis_label_font_size)
    plt.legend()

    if save_thumbnail:
        p = P.path_directory_set_result(set_name)
        image_name = f"variable_fitting.png"
        path = P.join(p, image_name)
        logging.info(f"Saving variable fitting plot to '{path}'.")
        plt.savefig(path, dpi=300)

    if not dont_show:
        plt.show()


def _plot_with_shadow(ax_obj, x_data, y_data, y_data_std, color, label, ls='-') -> None:
    """Plot data with standard deviation as shadow.

    Data must be sorted to show correctly.

    :param ax_obj:
        Pyplot axes object to plot to.
    :param x_data:
        Data x values (wavelengths).
    :param y_data:
        Data y values as numpy.array.
    :param y_data_std:
        Standard deviation as numpy array. The shadow is drawn as +- std/2.
    :param color:
        Color of the plot and shadow.
    :param label:
        Label of the value.
    :param ls:
        Line style. See pyplot linestyle documentation.
    """

    ax_obj.fill_between(x_data, y_data-(y_data_std/2), y_data+(y_data_std/2), alpha=alpha_error, color=color)
    ax_obj.plot(x_data, y_data, color=color, ls=ls, label=label)


def _plot_refl_tran_to_axis(axis_object, refl, tran, x_values, x_label, invert_tran=False, refl_color=color_reflectance, tran_color=color_transmittance):
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
    marker = '.'
    axis_object.plot(x_values, refl, label="Reflectance", color=refl_color, marker=marker)
    axt.plot(x_values, tran, label="Transmittance", color=tran_color, marker=marker)

    axis_object.set_ylim([0, 1])
    if invert_tran:
        axt.set_ylim([1, 0])
    else:
        axt.set_ylim([0, 1])

"""
This file contains plotting-related code.

Tips for plotting:
https://towardsdatascience.com/5-powerful-tricks-to-visualize-your-data-with-matplotlib-16bc33747e05

"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

from src import constants as C
from src.data import file_handling as FH, toml_handling as TH, file_names as FN, path_handling as PH
from src.leaf_model import nn, surf, training_data as training, surface_functions
from src.utils import data_utils as DU, spectra_utils as SU


figsize = (12,6)
"""Figure size for two plot figures."""
figsize_single = (7,6)
"""Figure size for single plot figures."""
fig_title_font_size = 18
"""Title font size."""
axis_label_font_size = 16
"""Axis label font size"""
save_resolution = 600
"""Save resolution for plots in dots per inch."""

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


def plot_default_soil_visualization(wls, reflectances, labels, save=True, dont_show=True):
    """Plot visualization (reflectances) of the default soil types.

    :param wls:
        List of wavelengths [400,401,...2500].
    :param reflectances:
        2D-array of reflectances. One row for one soil type.
    :param labels:
        List of soil names to be used as labels. These are also used
        to infer plot colors.
    :param save:
        If True (default), save image to disc.
    :param dont_show:
        If True (default), the plot is not shown, otherwise, show the plot using
        pyplot show(), which will halt the execution of the program until the window
        is manually shut.
    """

    plt.close('all')
    cm_clay = cm.get_cmap('Greys')
    cm_sand = cm.get_cmap('Oranges')
    cm_peat = cm.get_cmap('Greens')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.suptitle(f"Default soils", fontsize=fig_title_font_size)

    for i, reflectance in enumerate(reflectances):
        # Select color map and value based on the label
        if 'wet' in labels[i]:
            selector = 0.7
        if 'humid' in labels[i]:
            selector = 0.5
        if 'dry' in labels[i]:
            selector = 0.3
        if 'clay' in labels[i]:
            cm_active = cm_clay
        if 'sand' in labels[i]:
            cm_active = cm_sand
        if 'peat' in labels[i]:
            cm_active = cm_peat

        ax.plot(wls, reflectance, lw=1., label=labels[i], c=cm_active(selector))

    ax.set_xlabel('Wavelength [nm]', fontsize=axis_label_font_size)
    ax.set_ylabel(f'Reflectance', fontsize=axis_label_font_size)
    plt.legend()

    if save:
        folder = PH.path_directory_soil_data()
        image_name = "default_soils." + image_type
        path = PH.join(folder, image_name)
        logging.info(f"Saving default soil reflectance plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if not dont_show:
        plt.show()


def plot_blender_soil(wls, reflectances, soil_name, wls_resampled=None, reflectances_resampled=None, forest_id=None, dont_show=True, save=True):
    """Plots possibly resampled soil reflectance.

    :param wls:
        Original wavelengths as a list.
    :param reflectances:
        Original soil reflectances.
    :param soil_name:
        Name of the soil used. This will be shown in the plot and in resulting file's name.
    :param wls_resampled:
        Optional, resampled wavelengths.
    :param reflectances_resampled:
        Optional, resampled reflectances corresponding to resampled wavelengths.
    :param forest_id:
        Forest id is needed if save=True.
    :param dont_show:
        If True (default), the plot is not shown, otherwise, show the plot using
        pyplot show(), which will halt the execution of the program until the window
        is manually shut.
    :param save:
         If True (default), save image to forest scene directory (forest_id must then be given too).

    :raises
        AttributeError if save=True but forest_id=None.
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.suptitle(f"Resampled soil reflectance '{soil_name}'", fontsize=fig_title_font_size)

    ax.plot(wls, reflectances, lw=1., label=soil_name, c='grey')

    if wls_resampled is not None and reflectances_resampled is not None:
        ax.plot(wls_resampled, reflectances_resampled, lw=2., c='green')

    if save:
        if forest_id is None:
            raise AttributeError(f"Saving soil reflectance requested but no forest id was given to define proper path.")
        directory = PH.path_directory_forest_scene(forest_id=forest_id)
        image_name = f"soil_reflectance_{soil_name}{C.postfix_plot_image_format}"
        path = PH.join(directory, image_name)
        logging.info(f"Saving blender soil plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if not dont_show:
        plt.show()


def plot_reflectance_lab(HSV_value, reflectance, powers, plot_name=None, show=False, save=True):
    """Plot simulation result of virtual reflectance lab.

    :param HSV_value:
        Used hue-saturation-value values as a list of floats.
    :param reflectance:
        List of list of reflectances. Length of the first list must match the sun powers used.
    :param powers:
        List of sun powers.
    :param plot_name:
        Name for the plot file.
    :param show:
        If `True`, show interactive plot. Default is False.
    :param save:
        If `True`, save plot to project root directory. Default is True.
    :raises ValueError:
        If `len(powers) != len(reflectance)`.
    """

    if len(powers) != len(reflectance):
        raise ValueError(f"Length of sun powers {len(powers)} and measurements {len(reflectance)} must match.")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)

    for i,measurement in enumerate(reflectance):
        ax.set_xlabel('HSV value', fontsize=axis_label_font_size)
        ax.set_ylabel(f'Reflectance', fontsize=axis_label_font_size)
        ax.plot(HSV_value, measurement, label=f'Sun power {powers[i]} [W/m2]')

    plt.legend()

    if save:
        path = PH.join(C.path_project_root, f"{plot_name}{C.postfix_plot_image_format}")
        logging.info(f"Saving reflectance power plot to project root '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if show:
        plt.show()


def plot_light_data(wls, irradiances, wls_binned=None, irradiances_binned=None, forest_id=None, sun_plot_name=None, show=False, lighting_type='sun'):
    """Plot used sun or sky data for a scene.

    Plot can either be shown or saved. Plot is saved if scene_id is given.

    Three main uses:
        1. just original spectrum
        2. just binned spectrum
        3. both

    :param wls:
        Either original (1 nm resolution) or binned wavelengths.
    :param irradiances:
        Irradiances corresponding to wls in [W/m2/nm].
    :param wls_binned:
        Optional. Binned (integrated) wavelengths over certain bandwidth. Used
        bandwidth is calculated by wls_binned[1] - wls_binned[0].
    :param irradiances_binned:
        Optional. Binned (integrated) irradiances correcponding to wls_binned.
        Both must be given so that binned irradiances can be plotted.
    :param forest_id:
        Optional. Forest id for saving the plot to the scene directory.
    :param sun_plot_name:
        Optional. Sun filename for naming the image file.
    :param show:
        If True, the plot is shown to the user. Default is False.
    :param lighting_type:
        String either 'sun' or 'sky'.
    """

    bandwith = wls[1] - wls[0]
    if wls_binned is not None and irradiances_binned is not None:
        bandwith_binned = wls_binned[1] - wls_binned[0]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.suptitle(f"{lighting_type} spectrum", fontsize=fig_title_font_size)

        ax[0].plot(wls, irradiances, label=f'{lighting_type} 1 nm')
        ax[0].set_title('Spectra in file')
        ax[0].set_xlabel('Wavelength [nm]', fontsize=axis_label_font_size)
        ax[0].set_ylabel('Irradiance [W/m2/nm]', fontsize=axis_label_font_size)

        if lighting_type == 'sun':
            resampled_label = 'Resampled and normalized'
        elif lighting_type == 'sky':
            resampled_label = 'Resampled and normalized with sun'
        else:
            raise ValueError(f"Wrong lighting type. Expected lighting type either 'sun' or 'sky', was '{lighting_type}'.")

        ax[1].plot(wls_binned, irradiances_binned, label=f'Bandwidth {bandwith_binned:.0f} nm', alpha=0.5)
        ax[1].set_title(resampled_label)
        ax[1].set_xlabel('Wavelength [nm]', fontsize=axis_label_font_size)
        ax[1].set_ylabel(f'Irradiance [W/m2/{bandwith_binned:.0f}nm]', fontsize=axis_label_font_size)
        ax[1].set_xlim([wls[0],wls[-1]])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
        fig.suptitle(f"{lighting_type} spectrum", fontsize=fig_title_font_size)
        ax.set_xlabel('Wavelength [nm]', fontsize=axis_label_font_size)
        ax.set_ylabel(f'{lighting_type} irradiance [W/m2/{bandwith:.0f}nm]', fontsize=axis_label_font_size)
        ax.plot(wls, irradiances, label=f'Bandwidth {bandwith:.0f} nm')

    plt.legend()

    if sun_plot_name is None and lighting_type == 'sun':
        sun_plot_name = C.file_default_sun
    if sun_plot_name is None and lighting_type == 'sky':
        sun_plot_name = C.file_default_sky

    if forest_id is not None:
        path = PH.join(PH.path_directory_forest_scene(forest_id), f"{sun_plot_name.rstrip('.txt')}.png")
        plt.savefig(path, dpi=save_resolution)
    if show:
        plt.show()


def plot_nn_train_history(train_loss, test_loss, best_epoch_idx, dont_show=True, save_thumbnail=True,
                          file_name="nn_train_history.png") -> None:
    """Plot training history of neural network.

    :param train_loss:
        List of training losses (per epoch).
    :param test_loss:
        List of test losses (per epoch).
    :param best_epoch_idx:
        Index of best epoch for highlighting.
    :param dont_show:
        If true, does not show the interactive plot (that halts excecution). Always use True when
        running multiple times in a loop (hyperparameter tuning). Default True.
    :param save_thumbnail:
        If True, save plot to disk. Default True.
    :param file_name:
        Filename for saving the plot. Postfix '.png' is added if missing. Default name is "nn_train_history.png".
    :return:
    """

    if not file_name.endswith(C.postfix_plot_image_format):
        file_name = file_name + C.postfix_plot_image_format

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Training history", fontsize=fig_title_font_size)
    ax.plot(train_loss, label="Training loss")
    ax.plot(test_loss, label="Test loss")
    ax.scatter(best_epoch_idx, test_loss[best_epoch_idx], facecolors='none', edgecolors='r')
    ax.set_xlabel('Epoch', fontsize=axis_label_font_size)
    ax.legend()

    if save_thumbnail:
        folder = PH.path_directory_surface_model()
        # image_name = "nn_train_history.png"
        if not file_name.endswith(".png"):
            file_name = file_name + '.png'
        path = PH.join(folder, file_name)
        logging.info(f"Saving NN training history to '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)


def plot_trained_leaf_models(set_name='training_data', save_thumbnail=True, show_plot=False, plot_surf=True,
                             plot_nn=True, plot_points=True, nn_name='nn_default'):

    def variable_name_to_latex(v):
        """Change variable name into Latex format."""

        if v == 'ad':
            return r'$\rho_a$'
        elif v == 'sd':
            return r'$\rho_s$'
        elif v == 'ai':
            return r'$\alpha$'
        elif v == 'mf':
            return r'$\beta$'
        else:
            return v

    ad_train, sd_train, ai_train, mf_train, r_train, t_train, re_train, te_train = training.get_training_data(set_name=set_name)
    ad_train, sd_train, ai_train, mf_train, r_train, t_train = training.prune_training_data(ad_train, sd_train, ai_train, mf_train, r_train, t_train, re_train, te_train)
    train_params = [ad_train, sd_train, ai_train, mf_train]
    leaf_param_names = ['ad', 'sd', 'ai', 'mf']

    if plot_surf and surf.exists():
        ad_surf, sd_surf, ai_surf, mf_surf = surf.predict(r_train, t_train)
        surf_params = [ad_surf, sd_surf, ai_surf, mf_surf]
    else:
        surf_params = None

    if plot_nn and nn.exists(nn_name=nn_name):
        ad_nn, sd_nn, ai_nn, mf_nn = nn.predict(r_train, t_train, nn_name=nn_name)
        nn_params = [ad_nn, sd_nn, ai_nn, mf_nn]
    else:
        nn_params = None

    for i in range(4):

        # setup figure object
        fig = plt.figure(figsize=figsize_single)
        ax = plt.axes(projection="3d")
        ax.set_xlabel('R', fontsize=axis_label_font_size)
        ax.set_ylabel('T', fontsize=axis_label_font_size)
        ax.elev = 30
        # ax.azim = 225
        ax.azim = 180

        ax.set_zlabel(variable_name_to_latex(leaf_param_names[i]), fontsize=axis_label_font_size)

        if plot_points:
            # ax.scatter(r_train, t_train, train_params[i], marker='.', color='grey', alpha=0.1)
            train_surf = ax.plot_trisurf(r_train, t_train, train_params[i], label='training_data', alpha=0.3)
            train_surf._edgecolors2d = train_surf._edgecolor3d
            train_surf._facecolors2d = train_surf._facecolor3d

        if surf_params is not None:
            # surf_surf = ax.plot_trisurf(r_train, t_train, surf_params[i], linewidth=0.1, antialiased=True, color='red', alpha=0.2, label='surf', shade=True)
            surf_surf = ax.plot_trisurf(r_train, t_train, surf_params[i], label='surf')
            surf_surf._edgecolors2d = surf_surf._edgecolor3d
            surf_surf._facecolors2d = surf_surf._facecolor3d
        else:
            logging.warning(f"Cannot plot surface model plot as the model could not be used.")
        if nn_params is not None:
            # nn_surf = ax.plot_trisurf(r_train, t_train, nn_params[i], linewidth=0.1, antialiased=True, color='blue', alpha=0.2, label='nn', shade=True)
            nn_surf = ax.plot_trisurf(r_train, t_train, nn_params[i],  label='nn')
            nn_surf._edgecolors2d = nn_surf._edgecolor3d
            nn_surf._facecolors2d = nn_surf._facecolor3d
        else:
            logging.warning(f"Cannot plot nn model plot as the model could not be used.")

        if surf_params is not None and nn_params is not None:
            # legend breaks if not manually set as above. This is a known bug in matplotlib.
            ax.legend()

        if save_thumbnail:
            folder = PH.path_directory_surface_model()
            image_name = f"{leaf_param_names[i]}.png"
            path = PH.join(folder, image_name)
            logging.info(f"Saving surface plot to '{path}'.")
            plt.savefig(path, dpi=save_resolution)

        if show_plot:
            plt.show()


def plot_training_data_set(r_good, t_good, r_bad=None, t_bad=None, k1=None, b1=None, k2=None, b2=None, show=False, save=True):
    """Plot training data either interactively or save to disk.

    Constants k1,2 and b1,2 are used to visualize cutting lines along equation k*r + b.

    :param r_good:
        List of reflectances of good points.
    :param t_good:
        List of transmittances of good points.
    :param r_bad:
        List of reflectances of bad points.
    :param t_bad:
        List of transmittances of bad points.
    :param k1:
        First cutting line k.
    :param b1:
        First cutting line b.
    :param k2:
        Second cutting line k.
    :param b2:
        First cutting line b.
    :param show:
        Show interactive plot to user. Default is ```False```.
    :param save:
        Save plot to disk. Default is ```True```.
    """

    color_good = 'blue'
    color_bad = 'red'
    color_cut = 'black'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Training data", fontsize=fig_title_font_size)

    ax.scatter(r_good, t_good, c=color_good, alpha=0.5, marker='.', label='Good points')

    if r_bad is not None and t_bad is not None:
        ax.scatter(r_bad, t_bad, c=color_bad, alpha=0.5, marker='.', label='Bad points')
    else:
        logging.info("Badly fitted training data points were not given so they are not plotted.")

    if k1 and b1:
        x = np.array([-0.01,0.1])
        y = x*k1 + b1
        plt.plot(x, y, c=color_cut, linewidth=1)

    if k2 and b2:
        x = np.array([-b2,0.5])
        y = x*k2 + b2
        ax.plot(x, y, c=color_cut, linewidth=1)

    ax.set_xlabel('R', fontsize=axis_label_font_size)
    ax.set_ylabel('T', fontsize=axis_label_font_size)
    ax.legend()

    if save:
        folder = PH.path_directory_surface_model()
        image_name = "training_data" + C.postfix_plot_image_format
        path = PH.join(folder, image_name)
        logging.info(f"Saving the training data visualization plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if show:
        plt.show()


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

    subres_dict = TH.read_wavelength_result(set_name=set_name, wl=wl, sample_id=sample_id)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Optimization history (wl: {wl:.2f} nm)", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_ad])), subres_dict[C.key_wl_result_history_ad], label=C.key_wl_result_history_ad, color=color_ad)
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_sd])), subres_dict[C.key_wl_result_history_sd], label=C.key_wl_result_history_sd, color=color_sd)
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_ai])), subres_dict[C.key_wl_result_history_ai], label=C.key_wl_result_history_ai, color=color_ai)
    ax[0].plot(np.arange(len(subres_dict[C.key_wl_result_history_mf])), subres_dict[C.key_wl_result_history_mf], label=C.key_wl_result_history_mf, color=color_mf)
    ax[0].set_xlabel('Render call', fontsize=axis_label_font_size)
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)

    # Plot horizontal line to location of measured value
    x_data = np.arange(1, len(subres_dict[C.key_wl_result_history_r]))
    ax[1].plot(x_data, np.ones(len(x_data)) * subres_dict[C.key_wl_result_refl_measured], label=C.key_wl_result_refl_measured, color=color_history_target)
    ax[1].plot(x_data, 1 - np.ones(len(x_data)) * subres_dict[C.key_wl_result_tran_measured], label=C.key_wl_result_tran_measured, color=color_history_target)

    _plot_refl_tran_to_axis(ax[1], subres_dict[C.key_wl_result_history_r], subres_dict[C.key_wl_result_history_t], np.arange(len(subres_dict[C.key_wl_result_history_ai])), 'Render call', invert_tran=True)

    if save_thumbnail is not None:
        folder = PH.path_directory_subresult(set_name, sample_id)
        image_name = FN.filename_wl_result_plot(wl)
        path = PH.join(folder, image_name)
        logging.info(f"Saving the subresult plot to '{path}'.")
        plt.savefig(path, dpi=300) # keep this plot at low resolution as there may be quite a lot of these
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

    r = TH.read_set_result(set_name)
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
        folder = PH.path_directory_set_result(set_name)
        image_name = FN.filename_set_result_plot()
        path = PH.join(folder, image_name)
        logging.info(f"Saving the set result plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution, bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()


def plot_set_errors(set_name: str, dont_show=True, save_thumbnail=True):
    """Plots averaged optimization errors of a sample. """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Optimization errors ", fontsize=fig_title_font_size)
    marker = '.'

    ax.set_ylabel('RMSE', fontsize=axis_label_font_size)

    ids = FH.list_finished_sample_ids(set_name)
    wls = []
    refl_errs = []
    tran_errs = []
    for _,sample_id in enumerate(ids):
        result = TH.read_sample_result(set_name, sample_id)
        wls = result[C.key_sample_result_wls]
        refl_errs.append(result[C.key_sample_result_re])
        tran_errs.append(result[C.key_sample_result_te])

    wls = np.array(wls)
    refl_errs_mean = np.array(refl_errs).mean(axis=0)
    tran_errs_mean = np.array(tran_errs).mean(axis=0)
    refl_errs_std = np.array(refl_errs).std(axis=0)
    tran_errs_std = np.array(tran_errs).std(axis=0)

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
        folder = PH.path_directory_set_result(set_name)
        image_name = FN.filename_set_error_plot()
        path = PH.join(folder, image_name)
        logging.info(f"Saving the set error plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if not dont_show:
        plt.show()


def plot_resampling(set_name: str, dont_show=True, save_thumbnail=True) -> None:
    """Plots leaf resampled spectra along with the original for all leaf samples in given set.

    :param set_name:
        Set name.
    :param save_thumbnail:
        If True, a PNG image is saved. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    """

    target_ids = FH.list_target_ids(set_name=set_name)

    for sample_id in target_ids:
        target_original = TH.read_target(set_name=set_name, sample_id=sample_id, resampled=False)
        target_resampled = TH.read_target(set_name=set_name, sample_id=sample_id, resampled=True)
        wls_org, refl_org, tran_org = DU.unpack_target(target_original)
        wls_resampled, refl_resampled, tran_resampled = DU.unpack_target(target_resampled)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        fig.suptitle(f"Resampling of leaf sample {sample_id}", fontsize=fig_title_font_size)
        _plot_refl_tran_to_axis(axis_object=ax, refl=refl_org, tran=tran_org, x_values=wls_org, x_label='Wavelength [nm]', refl_color='black', tran_color='black', invert_tran=True)
        _plot_refl_tran_to_axis(axis_object=ax, refl=refl_resampled, tran=tran_resampled, x_values=wls_resampled, x_label='Wavelength [nm]', invert_tran=True)

        if save_thumbnail:
            folder = PH.path_directory_target(set_name=set_name)
            image_name = FN.filename_resample_plot(sample_id=sample_id)
            path = PH.join(folder, image_name)
            logging.info(f"Saving resampling plot to '{path}'.")
            plt.savefig(path, dpi=save_resolution)
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

    result = TH.read_sample_result(set_name, sample_id)
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
    ax[0].set_xlabel(x_label, fontsize=axis_label_font_size)
    ax[0].legend()
    ax[0].set_ylim(variable_space_ylim)
    _plot_refl_tran_to_axis(ax[1], result[C.key_sample_result_rm], result[C.key_sample_result_tm], result[C.key_sample_result_wls], x_label, invert_tran=True, refl_color='black', tran_color='black')
    _plot_refl_tran_to_axis(ax[1], result[C.key_sample_result_r], result[C.key_sample_result_t], result[C.key_sample_result_wls], x_label, invert_tran=True)
    if save_thumbnail:
        folder = PH.path_directory_set_result(set_name)
        image_name = FN.filename_sample_result_plot(sample_id=sample_id)
        path = PH.join(folder, image_name)
        logging.info(f"Saving the sample result plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution)
    if not dont_show:
        plt.show()


def replot_wl_results(set_name: str):
    """Replot wavelength results.

    Overwrites existing plots.
    """

    sample_ids = FH.list_finished_sample_ids(set_name)
    for sample_id in sample_ids:
        d = TH.read_sample_result(set_name, sample_id=sample_id)
        wls = d[C.key_sample_result_wls]
        for wl in wls:
            plot_wl_optimization_history(set_name, wl=wl, sample_id=sample_id)


def _plot_starting_guess_coeffs_fitting(dont_show=True, save_thumbnail=True, set_name: str = None) -> None:
    """Plot starting guess poynomial fit with data.

    Used only when generating the starting guess.

    :param set_name:
        Custom set name to fetch the data from. If not given, default set name variable
        'starting_guess_set_name' stored in constants.py is used.
    """

    plt.close('all')

    if set_name is None:
        set_name = C.starting_guess_set_name

    a_list, ad_list, sd_list, ai_list, mf_list = SU.get_starting_guess_points(set_name=set_name)

    ms = 10 # markersize
    ls = 2 # linesize
    plt.scatter(a_list, ad_list, label='Absorption density', color=color_ad, s=ms)
    plt.scatter(a_list, sd_list, label='Scattering density', color=color_sd, s=ms)
    plt.scatter(a_list, ai_list, label='Scattering anisotropy', color=color_ai, s=ms)
    plt.scatter(a_list, mf_list, label='Mix factor', color=color_mf, s=ms)

    coeffs = TH.read_starting_guess_coeffs()
    for _,key in enumerate(coeffs):
        coeff  = coeffs[key]
        y = np.array([np.sum(np.array([coeff[i] * (j ** i) for i in range(len(coeff))])) for j in a_list])
        plt.plot(a_list, y, color='black', linewidth=ls)

    plt.xlabel('Absorption', fontsize=axis_label_font_size)
    plt.ylabel('Material parameter', fontsize=axis_label_font_size)
    plt.legend()

    if save_thumbnail:
        p = PH.path_directory_set_result(set_name)
        image_name = f"variable_fitting.png"
        path = PH.join(p, image_name)
        logging.info(f"Saving variable fitting plot to '{path}'.")
        plt.savefig(path, dpi=save_resolution)

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

    axis_object.set_xlabel(x_label, fontsize=axis_label_font_size)
    axis_object.set_ylabel('Reflectance', color=refl_color, fontsize=axis_label_font_size)
    axis_object.tick_params(axis='y', labelcolor=refl_color)
    # Make twin axis for transmittance
    axt = axis_object.twinx()
    axt.set_ylabel('Transmittance', color=tran_color, fontsize=axis_label_font_size)
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

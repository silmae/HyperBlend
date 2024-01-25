import copy
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sphinx.cmd.quickstart import valid_dir

from src.data import path_handling as PH
from src import constants as C
from src import plotter
from src.algae import algae_utils as utils

"""

This directory contains samples used in algae validation tests. 
Samples are reflectances and transmittances measured with a 
spectrophotometer and an integrating sphere.

T for transmittance and R for reflectance

Algae 1 is more dense solution than Algae 2

Cell counter data:

Algae sample; cells/mL; mg/mL
A1 a ; 7.148E+07 ; 2.36
A1 b ; 5.826E+07 ; 2.31
A1 c ; 5.242E+07 ; 2.09
A2 a ; 6.216E+06 ; 0.1786
A2 b ; 5.626E+06 ; 0.1545
A2 c ; 6.463E+06 ; 0.2609

"""

dir_algae = "algae_measurement_sets"
dir_algae_measurement_set = '30_11_2023'
path_algae_measurement_set = PH.join(PH.path_directory_project_root(), dir_algae, dir_algae_measurement_set)

colors = ['r', 'g', 'b', 'c', 'm']
span = 10

sample_numbers_30_11_2023 = {
    '5208': 'White before',
    '5209': 'Growth media T',
    '5210': 'Growth media R',
    '5211': 'Algae 1 T',
    '5212': 'Algae 1 R',
    '5213': 'Algae 2 T',
    '5214': 'Algae 2 R',
    '5215': 'Dark',
    '5216': 'White after',
}

def plot_references(dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    fig.suptitle(f"References", fontsize=plotter.fig_title_font_size)
    ax[0].set_title('White')
    ax[1].set_title('Cuvette refs')

    wls, val_light_before = utils.read_sample(sample_nr='5208', measurement_set_name=path_algae_measurement_set)
    _, val_light_after = utils.read_sample(sample_nr='5216', measurement_set_name=path_algae_measurement_set)
    _, val_dark = utils.read_sample(sample_nr='5215', measurement_set_name=path_algae_measurement_set)
    _, val_gt = utils.read_sample(sample_nr='5209', measurement_set_name=path_algae_measurement_set)
    _, val_gr = utils.read_sample(sample_nr='5210', measurement_set_name=path_algae_measurement_set)

    ax[0].plot(wls, val_light_before, label=sample_numbers_30_11_2023['5208'])
    ax[0].plot(wls, val_light_after, label=sample_numbers_30_11_2023['5216'])
    # utils._plot_refl_tran_to_axis(ax[0], refl=val_er, tran=val_et, x_values=wls, x_label="Wavelength [nm]",
    #                               invert_tran=True, label='Empty cuvette', color='orange')

    # Normalize with white plug
    # val_et = val_et / val_light_before
    # val_er = val_er / val_light_before
    # val_bt = val_bt  / val_light_before
    # val_br = val_br  / val_light_before
    # val_dit = val_dit  / val_light_before
    # val_dir = val_dir  / val_light_before

    utils._plot_refl_tran_to_axis(ax[1], refl=val_gr, tran=val_gt, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Broth', color='brown')

    _, val_a1r = utils.read_sample(sample_nr='5212', measurement_set_name=path_algae_measurement_set)
    _, val_a2r = utils.read_sample(sample_nr='5214', measurement_set_name=path_algae_measurement_set)
    _, val_a1t = utils.read_sample(sample_nr='5211', measurement_set_name=path_algae_measurement_set)
    _, val_a2t = utils.read_sample(sample_nr='5213', measurement_set_name=path_algae_measurement_set)

    # Normalize with white plug
    # val_a1r = val_a1r / val_light_before
    # val_a2r = val_a2r / val_light_before

    # Smooth
    # val_a1r = utils.smooth_data_np_convolve(arr=val_a1r, span=span)
    # val_a2r = utils.smooth_data_np_convolve(arr=val_a2r, span=span)

    utils._plot_refl_tran_to_axis(ax[1], refl=val_a1r, tran=val_a1t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 1', color=colors[0])
    utils._plot_refl_tran_to_axis(ax[1], refl=val_a2r, tran=val_a2t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 2', color=colors[1])


    # ax[1].plot(wls, val_a1r, label="Algae 1", color=colors[0])
    # ax[1].plot(wls, val_a2r, label="Algae 2", color=colors[1])

    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)

    ylim = [0,1.0]
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)

    ax[0].legend()
    ax[1].legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"references.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution,bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()


def plot_algae(dont_show=True, save_thumbnail=True, ret_sampl_nr=1):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    # fig.suptitle(f"Algae", fontsize=plotter.fig_title_font_size)
    ax[0].set_title('Transmittance')
    ax[1].set_title('Reflectance')

    wls, val_light_before = utils.read_sample(sample_nr='5208', measurement_set_name=path_algae_measurement_set)
    _, val_light_after = utils.read_sample(sample_nr='5216', measurement_set_name=path_algae_measurement_set)
    _, val_dark = utils.read_sample(sample_nr='5215', measurement_set_name=path_algae_measurement_set)

    _, val_bt = utils.read_sample(sample_nr='5209', measurement_set_name=path_algae_measurement_set)
    _, val_br = utils.read_sample(sample_nr='5210', measurement_set_name=path_algae_measurement_set)

    _, val_a1t = utils.read_sample(sample_nr='5211', measurement_set_name=path_algae_measurement_set)
    _, val_a1r = utils.read_sample(sample_nr='5212', measurement_set_name=path_algae_measurement_set)
    _, val_a2t = utils.read_sample(sample_nr='5213', measurement_set_name=path_algae_measurement_set)
    _, val_a2r = utils.read_sample(sample_nr='5214', measurement_set_name=path_algae_measurement_set)

    # Spctral range
    low = 400
    high = 700
    val_light_before = utils.range(wls=wls, low=low, high=high, val=val_light_before)
    val_light_after = utils.range(wls=wls, low=low, high=high, val=val_light_after)
    val_dark = utils.range(wls=wls, low=low, high=high, val=val_dark)
    val_a1t = utils.range(wls=wls, low=low, high=high, val=val_a1t)
    val_a1r = utils.range(wls=wls, low=low, high=high, val=val_a1r)
    val_a2t = utils.range(wls=wls, low=low, high=high, val=val_a2t)
    val_a2r = utils.range(wls=wls, low=low, high=high, val=val_a2r)

    wls = utils.range(wls=wls, low=low, high=high, val=wls)

    # Normalize to lamp white
    final_ref_t = val_light_before

    # Attempt to normilize reflectance to empty cuvette
    # Does not really work as we would be dividing with a small number
    #   that explodes the final reflectance value, resulting in negative
    #   absorption
    # val_dir_mean = np.mean(val_dir)
    # val_er_mean = np.mean(val_er)
    # diff_we_r = val_dir / val_er
    # mean_we_r = np.mean(diff_we_r)
    # final_ref_r = val_er * mean_we_r

    # Normalize to lamp white
    final_ref_r = val_light_before

    # Reference

    np.clip(val_a1t, 0., 1.)
    np.clip(val_a1r, 0., 1.)
    np.clip(val_a2t, 0., 1.)
    np.clip(val_a2r, 0., 1.)

    val_a1t = np.clip(val_a1t / final_ref_t, 0., 1.)
    val_a1r = np.clip(val_a1r / final_ref_r, 0., 1.)
    val_a2t = np.clip(val_a2t / final_ref_t, 0., 1.)
    val_a2r = np.clip(val_a2r / final_ref_r, 0., 1.)

    # span2 = 3
    sigma = 5

    val_a1t_rough = copy.deepcopy(val_a1t)
    val_a2t_rough = copy.deepcopy(val_a2t)
    val_a1r_rough = copy.deepcopy(val_a1r)
    val_a2r_rough = copy.deepcopy(val_a2r)

    val_a1t = utils.smooth_data_np_convolve(arr=val_a1t, span=sigma)
    val_a2t = utils.smooth_data_np_convolve(arr=val_a2t, span=sigma)

    ax[0].plot(wls, val_a1t, label='Sample 1', color=colors[0])
    ax[0].plot(wls, val_a2t, label='Sample 2', color=colors[1])

    ax[0].plot(wls, val_a1t_rough, label='Sample 1 rough')
    ax[0].plot(wls, val_a2t_rough, label='Sample 2 rough')

    val_a1r = utils.smooth_data_np_convolve(arr=val_a1r,span=sigma)
    val_a2r = utils.smooth_data_np_convolve(arr=val_a2r,span=sigma)

    ax[1].plot(wls, val_a1r, label='Sample 1', color=colors[0])
    ax[1].plot(wls, val_a2r, label='Sample 2', color=colors[1])

    ax[1].plot(wls, val_a1r_rough, label='Sample 1 rough')
    ax[1].plot(wls, val_a2r_rough, label='Sample 2 rough')

    # utils._plot_refl_tran_to_axis(ax, refl=val_a1r, tran=val_a1t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 1', color=colors[0])
    # utils._plot_refl_tran_to_axis(ax, refl=val_a2r, tran=val_a2t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 2', color=colors[1])
    # utils._plot_refl_tran_to_axis(ax, refl=val_a3r, tran=val_a3t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 3', color=colors[2])
    # utils._plot_refl_tran_to_axis(ax, refl=val_a4r, tran=val_a4t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 4', color=colors[3])
    # utils._plot_refl_tran_to_axis(ax, refl=val_a5r, tran=val_a5t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Algae 5', color=colors[4])

    # ax[0].plot(wls_wt, percentage_t, label=f"Difference (mean {percentage_t_mean:.2f})")

    # ax[1].plot(wls_er, val_er, label=sample_numbers_01_09_2023['4791'])
    # ax[1].plot(wls_wr, val_wr, label=sample_numbers_01_09_2023['4791'])
    # ax[1].plot(wls_wr, percentage_r, label=f"Difference (mean {percentage_r_mean:.2f})")

    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)

    ylim = [0,1.0]
    ax[0].set_ylim(ylim)
    ax[1].set_ylim([0,1.0])

    ax[0].legend()
    ax[1].legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"algae.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution, bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()

    if ret_sampl_nr == 1:
        return wls, val_a1r, val_a1t
    elif ret_sampl_nr == 2:
        return wls, val_a2r, val_a2t
    else:
        logging.warning(f"Unsupported sample number {ret_sampl_nr}")
        pass


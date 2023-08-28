"""

Read results of spectrofotometer measurements.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.data import path_handling as PH
from src import constants as C
from src import plotter

dir_algae = "algae_samples"
dir_algae_measurement_set = '24_08_2023'
path_algae_measurement_set = PH.join(PH.path_directory_project_root(), dir_algae, dir_algae_measurement_set)

sample_numbers = {
    4771: "Empty",
    4772: "Kyvette empty trans",
    4773: "Kyvette empty refl",
    4774: "Kyvette water trans",
    4775: "Kyvette water refl",
    4776: "Algae 1 trans",
    4777: "Algae 1 refl",
    4778: "Algae 2 trans",
    4779: "Algae 2 refl",
    4780: "Algae 2 refl autowhite",
    4781: "Algae 1 refl autowhite",
    4782: "Algae 1 trans autowhite",
    4783: "Algae 2 trans autowhite",
}


def plot_manual_algae(dont_show=True, save_thumbnail=True):
    wls, val_a1t = read_sample('4776')
    _, val_a1r = read_sample('4777')
    _, val_a2t = read_sample('4778')
    _, val_a2r = read_sample('4779')

    # Empty kyvettes
    _, refer_t = read_sample('4772')
    _, refer_r = read_sample('4773')

    # Water-filled
    # _, refer_t = read_sample('4774')
    # _, refer_r = read_sample('4775')

    percentage_t, percentage_r = get_water_empty_diff()

    refer_t = refer_t * percentage_t
    refer_r = refer_r * percentage_r

    # refer_t = refer_t * 1.05
    # refer_r = refer_r * 1.27

    val_a1t = val_a1t / refer_t
    val_a1r = val_a1r / refer_r
    val_a2t = val_a2t / refer_t
    val_a2r = val_a2r / refer_r

    # Smoothing
    span = 10
    val_a1t = smooth_data_np_convolve(val_a1t, span)
    val_a1r = smooth_data_np_convolve(val_a1r, span)
    val_a2t = smooth_data_np_convolve(val_a2t, span)
    val_a2r = smooth_data_np_convolve(val_a2r, span)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    fig.suptitle(f"Algae with manual corrections", fontsize=plotter.fig_title_font_size)

    plotter._plot_refl_tran_to_axis(ax[0], refl=val_a1r, tran=val_a1t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True)
    plotter._plot_refl_tran_to_axis(ax[1], refl=val_a2r, tran=val_a2t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True)

    ax[0].set_title('Algae 1')
    ax[1].set_title('Algae 2')

    plt.legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"algae_manual_correction.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution)
    if not dont_show:
        plt.show()


def plot_ready_algae(dont_show=True, save_thumbnail=True):
    wls, val_a1t = read_sample('4782')
    _, val_a1r = read_sample('4781')
    _, val_a2t = read_sample('4783')
    _, val_a2r = read_sample('4780')

    val_a1t = val_a1t / 100
    val_a1r = val_a1r / 100
    val_a2t = val_a2t / 100
    val_a2r = val_a2r / 100

    percentage_t, percentage_r = get_water_empty_diff()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    fig.suptitle(f"Algae with automatic corrections", fontsize=plotter.fig_title_font_size)

    plotter._plot_refl_tran_to_axis(ax[0], refl=val_a1r, tran=val_a1t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True)
    plotter._plot_refl_tran_to_axis(ax[1], refl=val_a2r, tran=val_a2t, x_values=wls, x_label="Wavelength [nm]", invert_tran=True)

    ax[0].set_title('Algae 1')
    ax[1].set_title('Algae 2')

    plt.legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"algae_automatic_correction.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution)
    if not dont_show:
        plt.show()


def plot_water_empty_diff(dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    fig.suptitle(f"Empty vs. water filled kyvette", fontsize=plotter.fig_title_font_size)
    ax[0].set_title('Transmittance')
    ax[1].set_title('Reflectance')

    wls_light, val_light = read_sample('4771')
    wls_et, val_et = read_sample('4772')
    wls_er, val_er = read_sample('4773')
    wls_wt, val_wt = read_sample('4774')
    wls_wr, val_wr = read_sample('4775')

    val_et = val_et / val_light
    val_er = val_er / val_light
    val_wt = val_wt / val_light
    val_wr = val_wr / val_light

    percentage_t = 1 - (val_et / val_wt)
    percentage_r = 1 - (val_er / val_wr)
    percentage_t_mean = np.mean(percentage_t)
    percentage_r_mean = np.mean(percentage_r)

    ax[0].plot(wls_et, val_et, label=sample_numbers[4772])
    ax[0].plot(wls_wt, val_wt, label=sample_numbers[4774])
    ax[0].plot(wls_wt, percentage_t, label=f"Difference (mean {percentage_t_mean:.2f})")

    ax[1].plot(wls_er, val_er, label=sample_numbers[4773])
    ax[1].plot(wls_wr, val_wr, label=sample_numbers[4775])
    ax[1].plot(wls_wr, percentage_r, label=f"Difference (mean {percentage_r_mean:.2f})")

    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)

    ylim = [0,1.0]
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)

    ax[0].legend()
    ax[1].legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"empty_water.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution)
    if not dont_show:
        plt.show()


def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")


def get_water_empty_diff():

    wls_light, val_light = read_sample('4771')
    wls_et, val_et = read_sample('4772')
    wls_er, val_er = read_sample('4773')
    wls_wt, val_wt = read_sample('4774')
    wls_wr, val_wr = read_sample('4775')
    val_et = val_et / val_light
    val_er = val_er / val_light
    val_wt = val_wt / val_light
    val_wr = val_wr / val_light

    percentage_t = 1 - (val_et / val_wt)
    percentage_r = 1 - (val_er / val_wr)

    percentage_t += 1
    percentage_r += 1

    span = 10
    percentage_t = smooth_data_np_convolve(percentage_t, span)
    percentage_r = smooth_data_np_convolve(percentage_r, span)

    return percentage_t, percentage_r


def read_sample(sample_nr:str):
    filename = f"Sample{sample_nr}.Sample.asc"
    return read(filename)


def read(filename: str):

    p_file = PH.join(path_algae_measurement_set, filename)

    if not os.path.exists(path_algae_measurement_set):
        os.makedirs(path_algae_measurement_set)

    if not os.path.exists(p_file):
        raise FileNotFoundError(f"Could not find spectral algae measurement from '{p_file}'.")

    wls = []
    values = []

    with open(p_file, mode="r") as file:

        in_data = False

        for line in file.readlines():

            if line.startswith('#DATA'):
                in_data = True
                continue
            # else:
            #     print(f"Data line: '{line}'")

            if in_data:
                line = line.rstrip('\n')
                line = line.replace(',', '.')
                splitted = line.split('\t')
                wls.append(float(splitted[0]))
                values.append(float(splitted[1]))
                # print(splitted[0])


    return np.array(wls), np.array(values)

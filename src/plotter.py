
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import toml

from src import constants as C
from src import data_utils as DU
from src.render_parameters import RenderParametersForSeries
from src import toml_handlling as T
from src import file_handling as FH

figsize = (12,8)
fig_title_font_size = 18


def plot_list_variable_to_axis(axis_object, label: str, data, skip_first=False):
    length = len(data)
    if skip_first:
        axis_object.plot(np.arange(length - 1), data[1:length], label=label)
    else:
        axis_object.plot(np.arange(length), data, label=label)


def plot_x_line_to_axis(axis_object, label: str, data: float, x_values, invert=False):
    if invert:
        axis_object.plot(x_values, 1 - np.ones((len(x_values))) * data, label=label, color='red')
    else:
        axis_object.plot(x_values, np.ones((len(x_values))) * data, label=label, color='red')


def plot_refl_tran_to_axis(axis_object, refl, tran, x_values, x_label, invert_tran=False, skip_first=False):
    refl_color = 'blue'
    tran_color = 'orange'
    axis_object.set_xlabel(x_label)
    axis_object.set_ylabel('Reflectance', color=refl_color)
    axis_object.tick_params(axis='y', labelcolor=refl_color)
    # But use given x_values for plotting
    length = len(x_values)
    if skip_first:
        axis_object.plot(x_values[1:length], refl[1:length], label="Reflectance", color=refl_color)
    else:
        axis_object.plot(x_values, refl, label="Reflectance", color=refl_color)

    axt = axis_object.twinx()
    axt.set_ylabel('Transmittance', color=tran_color)
    axt.tick_params(axis='y', labelcolor=tran_color)
    if skip_first:
        axt.plot(x_values[1:length], tran[1:length], label="Transmittance", color=tran_color)
    else:
        axt.plot(x_values, tran, label="Transmittance", color=tran_color)

    axis_object.set_ylim([0, 1])
    if invert_tran:
        axt.set_ylim([1, 0])
    else:
        axt.set_ylim([0, 1])


def plot_subresult_opt_history(set_name: str, wl: float, save_thumbnail=False, dont_show=False):
    """Saves the image if savepath is given."""

    subres_dict = T.read_subresult(set_name=set_name, wl=wl)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Optimization history (wl: {wl:.2f} nm)", fontsize=fig_title_font_size)
    ax[0].set_title('Variable space')
    ax[1].set_title('Target space')
    plot_list_variable_to_axis(ax[0], C.subres_key_history_absorption_density,
                               subres_dict[C.subres_key_history_absorption_density], skip_first=True)
    plot_list_variable_to_axis(ax[0], C.subres_key_history_scattering_density,
                               subres_dict[C.subres_key_history_scattering_density], skip_first=True)
    plot_list_variable_to_axis(ax[0], C.subres_key_history_scattering_anisotropy,
                               subres_dict[C.subres_key_history_scattering_anisotropy], skip_first=True)
    plot_list_variable_to_axis(ax[0], C.subres_key_history_mix_factor,
                               subres_dict[C.subres_key_history_mix_factor], skip_first=True)
    ax[0].set_xlabel('iteration')
    ax[0].legend()
    plot_x_line_to_axis(ax[1], C.subres_key_reflectance_measured, subres_dict[C.subres_key_reflectance_measured],
                        np.arange(1,len(subres_dict[C.subres_key_history_reflectance])))
    plot_x_line_to_axis(ax[1], C.subres_key_transmittance_measured, subres_dict[C.subres_key_transmittance_measured],
                        np.arange(1,len(subres_dict[C.subres_key_history_transmittance])), invert=True)
    plot_refl_tran_to_axis(ax[1], subres_dict[C.subres_key_history_reflectance],
                           subres_dict[C.subres_key_history_transmittance],
                           np.arange(len(subres_dict[C.subres_key_history_scattering_anisotropy])),
                           'iteration', invert_tran=True,
                           skip_first=True)

    if save_thumbnail is not None:
        folder = FH.get_path_opt_result_plot(set_name)
        image_name = f"subresplot_wl{wl:.2f}.png"
        path = folder + '/' + image_name
        logging.info(f"Saving the subresult plot to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()


class Plotter:

    def __init__(self, rp: RenderParametersForSeries, x_label: str, x_values):
        self.rp = rp
        self.x_label = x_label
        self.x_values = x_values
        self.plot_folder = os.path.normpath(C.path_project_root + 'plot')
        x_ndvalues = np.array(x_values)
        self.filename = f"{x_label}_{x_ndvalues.min():.1f}-{x_ndvalues.max():.1f}"
        self.r = None
        self.t = None

        refl_list = []
        tran_list = []
        # Get data according to wavelengths
        for i, wl in enumerate(self.rp.wl_list):
            refl_list.append(DU.get_relative_refl_or_tran(C.imaging_type_refl, wl))
            tran_list.append(DU.get_relative_refl_or_tran(C.imaging_type_tran, wl))

        self.r = np.array(refl_list)
        self.t = np.array(tran_list)

    def dump(self):
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        d = self.rp.get_as_dict()
        d['x_label'] = self.x_label
        d['x_values'] = self.x_values
        d['reflectance'] = self.r
        d['transmittance'] = self.t
        path = os.path.normpath(f"{self.plot_folder}/{self.filename}.toml")
        with open(path, 'w') as f:
            toml.dump(d, f, encoder=toml.TomlNumpyEncoder())
        return d

    def plot_wl_series(self, invert_tran_y=True, save_thumbnail=True, silent=False):


        fig, axr = plt.subplots(figsize=figsize)
        plot_refl_tran_to_axis(axr, self.r, self.t, self.x_values, self.x_label, invert_tran=True)

        # axr.set_xlabel(self.x_label)
        # axr.set_ylabel('Reflectance', color=refl_color)
        # axr.tick_params(axis='y', labelcolor=refl_color)
        # # But use given x_values for plotting
        # axr.plot(self.x_values, self.r, label="Reflectance", color=refl_color)
        #
        # axt = axr.twinx()
        # axt.set_ylabel('Transmittance', color=tran_color)
        # axt.tick_params(axis='y', labelcolor=tran_color)
        # axt.plot(self.x_values, self.t, label="Transmittance", color=tran_color)

        # axr.set_ylim([0, 1])
        # if invert_tran_y:
        #     axt.set_ylim([1, 0])
        # else:
        #     axt.set_ylim([0, 1])
        # plt.legend()

        if save_thumbnail:
            if not os.path.exists(self.plot_folder):
                os.makedirs(self.plot_folder)
            path = os.path.normpath(f"{self.plot_folder}/{self.filename}.png")
            plt.savefig(path)

        if not silent:
            plt.show()



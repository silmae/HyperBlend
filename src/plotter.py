
import os

import numpy as np
import matplotlib.pyplot as plt
import toml

from src import constants as C
from src import data_utils as DU
from src.render_parameters import RenderParametersForSeries

class Plotter:

    def __init__(self, rp: RenderParametersForSeries, x_label: str, x_values):
        self.rp = rp
        self.x_label = x_label
        self.x_values = x_values
        self.plot_folder = os.path.normpath(C.project_root_path + 'plot')
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
        refl_color = 'blue'
        tran_color = 'orange'

        fig, axr = plt.subplots()

        axr.set_xlabel(self.x_label)
        axr.set_ylabel('Reflectance', color=refl_color)
        axr.tick_params(axis='y', labelcolor=refl_color)
        # But use given x_values for plotting
        axr.plot(self.x_values, self.r, label="Reflectance", color=refl_color)

        axt = axr.twinx()
        axt.set_ylabel('Transmittance', color=tran_color)
        axt.tick_params(axis='y', labelcolor=tran_color)
        axt.plot(self.x_values, self.t, label="Transmittance", color=tran_color)

        axr.set_ylim([0, 1])
        if invert_tran_y:
            axt.set_ylim([1, 0])
        else:
            axt.set_ylim([0, 1])
        # plt.legend()

        if save_thumbnail:
            if not os.path.exists(self.plot_folder):
                os.makedirs(self.plot_folder)
            path = os.path.normpath(f"{self.plot_folder}/{self.filename}.png")
            plt.savefig(path)

        if not silent:
            plt.show()

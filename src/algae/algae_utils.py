import os

import numpy as np

from src.data import path_handling as PH
from src import plotter


def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")


def read_sample(measurement_set_name:str, sample_nr:str):
    filename = f"Sample{sample_nr}.Sample.asc"
    return read(filename=filename, measurement_set_name=measurement_set_name)


def read(measurement_set_name:str, filename: str):

    p_file = PH.join(measurement_set_name, filename)

    if not os.path.exists(measurement_set_name):
        os.makedirs(measurement_set_name)

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



def _plot_refl_tran_to_axis(axis_object, refl, tran, x_values, x_label, invert_tran=False,
                            color=None, label=None):
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

    if color is None:
        color = 'black'

    axis_object.set_xlabel(x_label, fontsize=plotter.axis_label_font_size)
    axis_object.set_ylabel('Reflectance', fontsize=plotter.axis_label_font_size)
    axis_object.tick_params(axis='y',)
    # Make twin axis for transmittance
    axt = axis_object.twinx()
    axt.set_ylabel('Transmittance', fontsize=plotter.axis_label_font_size)
    axt.tick_params(axis='y',)
    # But use given x_values for plotting
    marker = '--'
    axis_object.plot(x_values, refl, color=color,  label=label)
    axt.plot(x_values, tran, color=color,  label=label)

    axis_object.set_ylim([0, 1])
    if invert_tran:
        axt.set_ylim([1, 0])
    else:
        axt.set_ylim([0, 1])


def range(wls, val, low, high):

    tru = (wls >= low) & (wls <= high)
    thing = val[tru]
    return thing


"""
    General Spectral Vectors (GSV) soil model. The code in this script is
    based on the code written by authors of
    "GSV: a general model for hyperspectral soil reflectance simulation", 2019.
"""


import numpy as np

from src.data import path_handling as PH
from src import plotter

wls = np.arange(400, 2501, 10)
"""Native hyperspectral wavelengths of GSV ranging from 400-2500 nm with 10 nm resolution."""

new_wls = np.arange(400, 2501, 1)
"""Wavelengths from 400-2500 nm with 1 nm resolution. 
Spectral resolution in HyperBlend is assumed to be 1 nm, so we 
linearly interpolate the native GSV resolution to 1 nm."""

GSV = np.vstack([np.loadtxt(PH.path_file_soil_dry_vector()), np.loadtxt(PH.path_file_soil_humid_vector())])
"""The general spectral vectors derived in the manuscript"""

default_soils = {
    "wet_clay":             [0.245, -0.039,  0.003, -0.145],
    "median_humid_clay":    [0.528, -0.011,  0.014, -0.129],
    "dry_clay":             [0.459,  0.011, -0.009,  0.038],
    "wet_sand":             [0.649, -0.065,  0.002, -0.587],
    "median_humid_sand":    [0.512, -0.044, -0.044, -0.103],
    "dry_sand":             [0.539, -0.009, -0.068,  0.079],
    "wet_peat":             [0.423, -0.163,  0.045, -0.461],
    "median_humid_peat":    [0.581, -0.268,  0.061, -0.303],
    "dry_peat":             [0.384, -0.220,  0.044, -0.066],
}
"""Default c_n and c_SM values for different kinds of soils from Table 3 in 
page 9 of the paper.
"""


def simulate_gsv_soil(c1: float, c2: float, c3: float, cSM: float):
    """Simulate soil spectra with GSV model.

    Parameters c1,c2,c3 are tuning parameters for dry soil (in descending order
    of importance). Parameter cSM is for wet soil. This allows arbitrary soil
    spectra generation. If you want to use one of the pre-defined soil types,
    use simulate_default_soil().

    :param c1:
        Tuning parameter for dry soil.
    :param c2:
        Tuning parameter for dry soil.
    :param c3:
        Tuning parameter for dry soil.
    :param cSM:
        Tuning parameter for wet soil. This is usually between -0.5 and 0.0.
        Positive values often causes water absorption peaks in near infrared to flip
        in the wrong way.
    :return:
        Returns reflectance spectra for given soil type and moisture defined by soil name.
        Reflectance is returned as a 1D Numpy array from 400-2500 nm with 1 nm resolution
        that is linearly interpolated from the native GSV resolution of 10 nm.
    """

    gsv_spectra = c1 * GSV[0] + c2 * GSV[1] + c3 * GSV[2] + cSM * GSV[3]
    resampled_spectra = np.interp(new_wls, wls, gsv_spectra)
    return resampled_spectra


def simulate_default_soil(soil_name: str):
    """Run GSV soil simulation with one of the default soil types (soil_name).

    :param soil_name:
        One of the following:
            "wet_clay"
            "median_humid_clay"
            "dry_clay"
            "wet_sand"
            "median_humid_sand"
            "dry_sand"
            "wet_peat"
            "median_humid_peat"
            "dry_peat"
    :return:
        Returns reflectance spectra for given soil type and moisture defined by soil name.
        Reflectance is returned as a 1D Numpy array from 400-2500 nm with 1 nm resolution
        that is linearly interpolated from the native GSV resolution of 10 nm.
    """

    soil_coeffs = default_soils[soil_name]
    spectra = simulate_gsv_soil(*soil_coeffs)
    return spectra


def visualize_default_soils(dont_show=True, save=True):
    """Visualize default soils' reflectance spectra.

    :param dont_show:
        If True (default), the plot is not shown, otherwise, show the plot using
        pyplot show(), which will halt the execution of the program until the window
        is manually shut.
    :param save:
        If True (default), save the plot to disk.
    """

    refls = []
    labels = []
    for key, item in default_soils.items():
        spec = simulate_gsv_soil(*item)
        refls.append(spec)
        labels.append(key)

    plotter.plot_default_soil_visualization(new_wls, reflectances=refls, labels=labels, dont_show=dont_show, save=save)

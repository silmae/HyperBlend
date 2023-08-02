import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as tk

from src.data import path_handling as PH
from src.utils import spectra_utils as SU
from src import plotter

'''
    General Spectral Vectors (GSV) soil model. The code in this script is 
    based on the code written by authors of 
    "GSV: a general model for hyperspectral soil reflectance simulation", 2019.
'''

# The hyperspectral wavelengths
wls = np.arange(400, 2501, 10)

new_wls = np.arange(400, 2501, 1)
# The general spectral vectors derived in the manuscript
GSV = np.vstack([np.loadtxt(PH.path_file_soil_dry_vector()), np.loadtxt(PH.path_file_soil_humid_vector())])
# The test hyperspectral data
# hyper = np.loadtxt('TestSpectrum.csv',delimiter=',',skiprows=1)
# The wavelengths of multispectral data
wvl = np.array([450,550,650,850,1650,2150])
# The test multispectral data sliced from hyperspectral data
# multi = hyper[np.in1d(WVL,wvl)]

'''
    This script shows how to simulate a series of soil spectra
    using combinations of four soil coefficients.
'''

"""
Default c_n and c_SM values for different kinds of soils from Table 3 in 
page 9 of the paper.
"""

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


def simulate_gsv_soil(c1: float, c2: float, c3: float, cSM: float):
    """Simulate soil spectra with GSV model.

    :param c1:
    :param c2:
    :param c3:
    :param cSM:
    :return:
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
    """

    soil_coeffs = default_soils[soil_name]
    spectra = simulate_gsv_soil(*soil_coeffs)
    return spectra


def visualize_default_soils():

    refls = []
    labels = []
    for key, item in default_soils.items():
        spec = simulate_gsv_soil(*item)
        refls.append(spec)
        labels.append(key)

    plotter.plot_default_soil_visualization(new_wls, reflectances=refls, labels=labels, dont_show=False, save=False)

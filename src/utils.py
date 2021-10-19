"""
Utility functions for all scripts in the project.

"""


import numpy as np
from numpy.polynomial import Polynomial


def parse_wl_from_image_name(image_name):
    """Parse the wavelength from an image name (filename).

    The name must be formed [refl|tran]_wl[0-9]+.[0-9]+/.*
    """

    tail = image_name.split("_wl", 1)[1]
    wl_s = tail.rsplit(".", 1)[0]
    return float(wl_s)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.

    Example
    -----
        original_list = [0,1,2,3,4,5,6,7,8,9]
        chunked_list = list(chunks(original_list, 3))
        -> [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fit_poly(x,y,degree,name):
    fit = Polynomial.fit(x, y, deg=degree, domain=[0, 1])
    coeffs = fit.convert().coef
    print(f"fitting coeffs for {name}: {coeffs}")
    # y = np.array([np.sum(np.array([coeffs[i] * (j ** i) for i in range(len(coeffs))])) for j in x])
    # plt.plot(x, y, color='black')
    return coeffs

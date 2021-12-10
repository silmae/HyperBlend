"""
Utility functions for all scripts in the project.

"""

from numpy.polynomial import Polynomial


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


def fit_poly(x, y, degree):
    """Fit a plynomial function to data.

    :param x:
        Data x values.
    :param y:
        Data y values.
    :param degree:
        Degree of the fuction to fit.
    :return:
        A list of coefficients.
    """

    fit = Polynomial.fit(x, y, deg=degree, domain=[0, 1])
    coeffs = fit.convert().coef
    return coeffs

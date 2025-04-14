import logging

import numpy as np


def get_x0():
    count = 4
    x0 = np.ones((count,)) * 0.1
    return x0


def safe_log(x):
    # Avoid division by zero warnings
    eps = 1e-10
    result = np.where(x > eps, x, -10)
    np.log(result, out=result, where=result > 0)
    return result


def function_exp(data, a, b, c, d):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    res = a*(np.exp(r*b)) + c*(np.exp(t*d))
    return res


def function_log(data, a, b, c, d):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    res = a*(safe_log(r*np.fabs(b))) + c*(safe_log(t*np.fabs(d)))
    return res


def function_polynomial(data, a, b, c, d):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    try:
        res = a*(r**b) + c*(t**d)
    # TODO useless when parallel. remove
    except RuntimeWarning as rw:
        raise RuntimeWarning(f"Divide by zero perhaps resulting from one of the fitting parameters "
                        f"being negative. Params are: [{a,b,c,d}]") from rw
    return res

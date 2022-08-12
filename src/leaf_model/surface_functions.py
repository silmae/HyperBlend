
import numpy as np


def get_x0():
    count = 4
    x0 = np.ones((count,)) * 0.1
    return x0


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
    res = a*(np.log(r*np.fabs(b))) + c*(np.log(t*np.fabs(d)))
    return res


def function_polynomial(data, a, b, c, d):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    res = a*(r**b) + c*(t**d)
    return res


import numpy as np


def function(data, a, b, c):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    # res = a * (r**b) * (t**c)
    res = a * (np.exp(-r*b)) * (np.exp(-t * c))
    return res

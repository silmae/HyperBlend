
import numpy as np


def get_x0():
    count = 8
    x0 = np.ones((count,)) * 0.1
    # x0 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    return x0


def function(data, a, b, c, d, e, f,g,h,):#i,j,k,l):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    # res = a*(r**b) + c*(t**d) #+ e*(r**f) + g*(t**h) + i*(r**j) + k*(t**l)
    res = a*(np.exp(r*b)) + c*(np.exp(t*d)) + e*(r**f) + g*(t**h)
    return res


def function2(data, a, b, c, d, e, f,g,h,):#i,j,k,l):
    """Function used in fitting parameter surface to reflectance-transmittance value pairs.

    Used in fitting and in retrieving Blender parameter values.
    """

    r = data[0]
    t = data[1]
    # res = a*(r**b) + c*(t**d) #+ e*(r**f) + g*(t**h) + i*(r**j) + k*(t**l)
    res = a*(np.exp(r*b)) + c*(np.exp(t*d)) #+ e*(r**f) + g*(t**h)
    return res

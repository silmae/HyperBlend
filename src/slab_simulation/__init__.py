"""

This package contains modules related to the slab model of HyperBlend.

There are three ways to convert reflectance and transmittance pairs (RT pairs)
into renderable slab material parameters:

1. opt: original optimization method that is very slow
2. material_param_surface: surface fitting method that is fast but rather inaccurate
3. material_param_neural: neural network method that is fast and slightly less accurate than optimization but much
    more accurate than surface fit. This is the recommended method.

"""

"""

This package contains modules related to the leaf model of HyperBlend.

There are three ways to convert reflectance and transmittance pairs (RT pairs)
into leaf material parameters:

1. material_param_optimization: original optimization method that is very slow
2. material_param_surface: surface fitting method that is fast but rather inaccurate
3. material_param_neural: neural network method that is fast and sligthly less accurate than optimization but much
more accurate than surface fit. This is the recommended method.

"""

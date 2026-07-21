"""
This module contains scripts that call Blender directly (prefix `bs_`) as well as
a few more common sub-modules that do not call Blender. All of these scripts
must be called using :py:mod:`rendering.blender_control` which acts as an
intermediary to decouple the HyperBlend's code from Blender calls.

.. note::
    Blender runs its own Python environment when called. That's why the
    import statements in all `bs_` files look a bit weird. The local HyperBlend
    modules have to be added to Blender's Python manually.

"""

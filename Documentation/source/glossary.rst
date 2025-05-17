
Glossary
============

This glossary contains some terms that are commonly used throughout the documentation.

Parameter Names
..................

Commonly used parameter names

.. glossary::

    forest_id
        Old name for :term:`system_sim_name`.

    system_sim_name
        Name of the system simulation. This is used to identify a specific
        simulation and used in the directory structure.

    set_name
        Old nane for :term:`slab_sim_name`.

    slab_sim_name
        Name of the slab simulation. This is used to identify a specific
        simulation and used in the directory structure.

    sample_id
        An old name for :term:`signal_id`.

    signal_id
        An integer that identifies a specific signal in a slab simulation.

    slab_model_name
        The name of a slab model. There can be many models trained for different
        purposes, such as slabs of different thicknesses. Each model has either an
        optimization, surface figging, or neural network solver, depending on how it
        was trained. The optimization solver is always usable even without training.


General
-------------

.. glossary::

    Slab simulation
        Slab simulation is the low level simulation of HyperBlend that represents a
        homogeneous slab of translucent material.

        In the context of plant simulation, a slab would represent a single leaf.
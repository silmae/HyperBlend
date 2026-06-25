.. _chap-glossary:

Glossary
============

This glossary contains some terms that are commonly used throughout the documentation.

General
-------------

.. glossary::

    Slab simulation
        Slab simulation is the low level simulation of HyperBlend that represents a
        homogeneous slab of translucent material.

        In the context of plant simulation, a slab would represent a single leaf.
        In other simulation context it might represent a slab of water full of
        microalgae as in :cite:`riihiaho25`.

    Solver
        Slab simulation solver which in code has a **name**:
        **opt** (optimization based), original solver, which is accurate but very slow.
        **nn** (neural network), which is fast (200 times faster than opt) with good accuracy
        (decrease of 2-4 times compared to opt). **surf** (surface fitting), which is worse
        than nn in accuracy but around the same speed.

    Signal
        Signal is a pair of reflectance and transmittance spectra in the
        :term:`Slab simulation`. In forestry application the signal is reflectance
        and transmittance of a single leaf. Signals can originate from real world
        measurements or they can be simulated with the integrated PROSPECT
        :cite:`feret17` leaf
        spectral simulator. In slab simulation, the signal is initially called
        `Target signal`. The point of the slab simulation is to solve which slab
        material parameter values (for Blender) produce the same reflectance and
        transmittance in the virtual environment. Solving of these parameters
        produces the `Result signal` that can be used in :term:`System simulation`.

    System simulation
        System simulation is the high level simulation where the material slabs
        of :term:`Slab simulation` will be placed to create a meaningful recreation of
        some macroscopic scale scene.

        In forest simulation, this system would be the forest (or a single tree)
        where the simulated slabs are placed. In simulating a production line of
        plastic parts, this could be the conveyor belt with feeding the plastic
        part forward in the process.

Parameter Names
..................

Commonly used parameter names

.. glossary::

    forest_id
        DEPRECATED Old name for :term:`system_sim_name`.

    system_sim_name
        Name of the system simulation. This is used to identify a specific
        simulation and used in the directory structure.

    set_name
        DEPRECATED Old nane for :term:`slab_sim_name`.

    slab_sim_name
        Name of the slab simulation. This is used to identify a specific
        simulation and used in the directory structure.

    sample_id
        DEPRECATED An old name for :term:`signal_id`.

    signal_id
        An integer that identifies a specific signal in a slab simulation.

    slab_model_name
        The name of a slab model. There can be many models trained for different
        purposes, such as slabs of different thicknesses. Each model has either an
        optimization, surface figging, or neural network solver, depending on how it
        was trained. The optimization solver is always usable even without training.

    runtime
        Runtime data that the program needs to perform all the tasks. This includes
        logging and Blender executable path, for example.
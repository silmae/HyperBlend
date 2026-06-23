.. _chap-slab-simulation:

The Slab Simulation
==============================

As already mentioned in the :doc:`beginner tutorial <basics>`, slab simulation
is the first separate simulation step, which creates a slab of material that
has certain spectral reflectance and transmittance properties. What these
properties are -- that is for the user to decide. This piece of tutorial
will dig into how they can be defined in HyperBlend from the practical point
of view. For the scientific foundation and details, the reader is encouraged
to take a look at the first two published HyperBlend papers,
i.e., :cite:`riihiaho22` and :cite:`riihiaho23`.


A bit more advanced target spectra
-------------------------------------------

Custom PROSPECT leaf spectra
"""""""""""""""""""""""""""""""""""""""

HyperBlend is totally agnostic to the source of the slab spectra. In the
beginner tutorial, we created a random reflectance and transmittance spectra
using PROSPECT simulator. However, the power of prospect is in that you can
define biochemical and (to some extent) biophysical variables of a
simulated leaf, so let's look how that is done through HyperBlend.

.. code-block:: python3

    from src.setup import initialization
    from src.slab_simulation import interface as SMI

    if __name__ == "__main__":

        runtime = initialization.initialize()

        slab_sim_name = "prospect_slab"
        SMI.generate_prospect_leaf(
            slab_sim_name=slab_sim_name,
            signal_id=0,
            n=1.5,
            ab=32.0,
            ar=8.0,
            brown=0.0,
            w=0.016,
            m=0.009,
            ant=0.0,
        )
        SMI.solve_slab_material_parameters(
            runtime=runtime,
            slab_sim_name=slab_sim_name,
            range_start=400,
            range_end=1000,
            resolution=100,
        )

The ``signal_id`` is zero by default, but if you create multiple leaf spectra
in a loop (as you probably would in any real application), just pass
the looping index as ``signal_id``
We simply called PROSPECT with desired carotenoid and chlorophyll (etc.)
content and solved again for slab material parameters. See the documentation
of :py:func:`~slab_model.interface.generate_prospect_leaf` for the explanation
of parameter names.


Using real-world leaf spectra
"""""""""""""""""""""""""""""""""""""""

We can also use real-world measurements as a source. The example below shows
how to use a custom target as target data and how to save and solve the slab
parameters:

.. code-block:: python3

    from src.setup import initialization
    from src.slab_simulation import interface as SMI
    from src.data import toml_handling as TH

    if __name__ == "__main__":

        runtime = initialization.initialize()
        set_name = "try_manual_set"

        # Example data list of lists where inner list holds the data
        target = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]

        # Write data to disk in a format the HyperBlend can understand
        TH.write_target(set_name, target, signal_id=0)

        # Solve as before
        SMI.solve_slab_material_parameters(runtime=runtime, slab_sim_name=set_name)

In this toy example we have only two spectral bands defined as ``target``.
If you have your measured data, for example, in a .csv file, you will have to
read it yourself into the list structure, and then it will be compliable with
the rest of HyperBlend code. Internally, HyperBlend saves most of non-image data
as .toml files. (Only files read by Blender scripts are saved as CSV.) That is
the reason we introduce one new import: `from src.data import toml_handling as TH`.
The :py:mod:`data.toml_handling` handles all (or at least most) writing and
reading of toml files.


Slab Models
-------------------------------------------

As explained in paper :cite:`riihiaho23`, HyperBlend has three slab models
one can use for solving. In the examples above, we used the default one
which is a neural network based solution.

...

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

The slab simulation consists of the actual simulations that are done by using
various :term:`Solver` s, training new solvers, and generating/reading/writing
slab spectra. Since we already touched the topic of spectra generation in the
beginner tutorial, we dig a bit deeper into that first, and then cover the
other topics.

The slab simulation is based on a virtual "measurements" of a slab's reflectance
and transmittance. This is based on a Blender scene
at `root/Internal/slab_sim_template.blend`


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
of :py:func:`~slab_simulation.interface.generate_prospect_leaf` for the explanation
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
        slab_sim_name = "try_manual_set"

        # Example data list of lists where inner list holds the data
        target = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]

        # Write data to disk in a format the HyperBlend can understand
        TH.write_target(slab_sim_name, target, signal_id=0)

        # Solve as before
        SMI.solve_slab_material_parameters(runtime=runtime, slab_sim_name=slab_sim_name)

In this toy example we have only two spectral bands defined as ``target``.
If you have your measured data, for example, in a .csv file, you will have to
read it yourself into the list structure, and then it will be compliable with
the rest of HyperBlend code. Internally, HyperBlend saves most of non-image data
as .toml files. (Only files read by Blender scripts are saved as CSV.) That is
the reason we introduce one new import: `from src.data import toml_handling as TH`.
The :py:mod:`data.toml_handling` handles all (or at least most) writing and
reading of toml files.


Slab Simulation Solvers
----------------------------

As explained in paper :cite:`riihiaho23`, HyperBlend has three slab simulation
:term:`Solver` s that can be used for solving the renderable slab material parameters.
In the examples above, we used the default one which is a neural network based
solution. It arguably offers the best balance between precision and speed.

For advanced users who want to simulate something else than the available forest
scenes, or who want to develop the software further, it may be necessary to
retrain the solvers. As the training data should be generated using
the original optimization based solver (slowest but most accurate of the ones
offered), we cover some of those topics now.

So, the three available slab solvers are:

    * **opt** (optimization based), original solver. Accurate but very slow
    * **nn** (neural network), which is fast (200 times faster than opt) with good accuracy (decrease of 2-4 times compared to opt)
    * **surf** (surface fitting), which is worse than nn in accuracy but around the same speed

The bolded names are how the solvers are called inside the code (as strings).
The modules they are coded in are named accordingly:
:py:mod:`slab_simulation.opt`, :py:mod:`slab_simulation.nn`, and :py:mod:`slab_simulation.surf`.
In general, you don't need to deal with these modules directly, unless you are a developer.

If we would like to solve the previous example using the `opt` solver, we would call it like this:

.. code-block:: python3

    SMI.solve_slab_material_parameters(runtime=runtime, slab_sim_name=slab_sim_name, solver='opt')

For using the surface
fitting method, we would call it with argument ``solver='surf'``.  Calling the solver is
still a bit more complicated than this because one may have trained multiple sets of solvers.
The call we use above will select the desired default solver that are available in the repository.

If you train custom solvers (we'll cover how to, next), they will reside under `root/Slab solvers/`.
Say, you trained a new set of solvers called ``'My own solvers'``. Then
you would call the :py:func:`~slab_simulation.interface.solve_slab_material_parameters`
like so

.. code-block:: python3

    SMI.solve_slab_material_parameters(
        runtime=runtime,
        slab_sim_name=slab_sim_name,
        range_start=400,
        range_end=1000,
        resolution=100,
        solver="nn",
        solver_dirname="My own solvers",
    )


Training custom solvers
---------------------------

Training new solvers comes into question if:

    - you want to improve the existing default ones
    - you need different slab thickness or do other changes to the slab simulation environment

The first case is more about if you want to do something or not. The default solvers
are pretty good, but they can definitely be improved. In the second case, if you introduced some
changes to the slab simulation environment (i.e., you are a developer), you **must** retrain
the solvers as they will no longer give correct results (although they may look like doing
sensible things).

.. note::
    The opt solver does not rely on training but it does need/benefit from a good starting
    guess of where to start the optimization process.

Training data
""""""""""""""""""

In :py:mod:`slab_simulation.interface`, there are two functions related to training:
:py:func:`slab_simulation.interface.train_models` and
:py:func:`slab_simulation.interface.iterative_train`. The latter essentially runs the
former with specific settings iteratively for a set number of times.
Let's take a look how to call the former one who does the training data generation
and the actual training with one call. We'll take it step by step as the function
takes a lot of attributes.

.. code-block:: python3

    if __name__ == "__main__":
        runtime = initialization.initialize()
        SMI.train_models(runtime=runtime, slab_sim_name_for_training='My training data', generate_data=True, dry_run=True)

This will generate a set of new training data with the default options. Except,
because we set ``dry_run=True``, it only prints out how many data points it would
have created without actually creating any. Let's change the way the training data will be
generated with some arguments

.. code-block:: python3

    SMI.train_models(
        runtime=runtime,
        slab_sim_name_for_training="My training data",
        generate_data=True,
        dry_run=True,
        starting_guess_type="curve",
        similarity_rt=0.25,,
        train_points_per_dim=20
    )

As stated in the API documentation for argument ``starting_guess_type``:
*String, one of 'hard-coded', 'curve', 'surf' in order of increasing complexity.
Hard-coded is only needed if training the other methods from absolute scratch (for
example if leaf material parameter count or bounds change in future development).
Curve fitting 'curve' is the method presented in the first HyperBlend paper*
:cite:`riihiaho22`. *It will
only work in cases where R and T are relatively close to each other (around +- 0.2).
Surface fitting method 'surf' can be used after the first training iteration has been carried
out. It can more robustly adapt to situations where R and T are dissimilar.*

The similarity of R (reflectance) and T (Transmittance) argument ``similarity_rt``
are discussed to some extent
in all papers published on HyperBlend and it is a very much a subject hindering the
accuracy and performance of HyperBlend. **So, if you are a developer, I hope you can
solve this problem some day.** Roughly, this *RT similarity requirement* means that
if R and T are far away from each other (e.g. R=0.05 and T=0.95 as they live in
interval 0-1), the slab model is incapable of producing good, reliable, or consistent
results.

``train_points_per_dim=20`` tell to how many parts the R and T dimensions are split.
In other words, with value 20, there will be 20 different values for R and 20 for T.

Actual training
"""""""""""""""""""""

So, let's assume we have some training data now generated and move to the training.
The training and data generation are very intertwined, so separating them here is
a bit forced, but let's try anyways.

For training, we will still use the same call but with different argument values.

.. code-block:: python3
    :linenos:

    SMI.train_models(
        runtime=runtime,
        slab_sim_name_for_training="My training data",
        generate_data=False,
        dry_run=False,
        train_surf=True,
        train_nn=True,
        layer_count=5,
        layer_width=1000,
        epochs=300,
        batch_size=32,
        learning_rate=0.01,
        patience=30,
        split=0.1,
        show_plot=False,
        solver_name_to_save=None,
        solver_name_to_use=None,
    )

Now we set ``generate_data=False``, since we already generated the data under that name.
We set ``train_surf=True`` and ``train_nn=True``, so they get trained. The rest are mostly
parameters for the training. The ``layer_count``, ``layer_width``, ``epochs``,
``batch_size``, ``learning_rate``, ``patience``, and ``split`` are related to NN training.
Surface fitting training does not have exposed arguments, so if you want to change something,
you will have to dig into its source code :py:mod:`slab_simulation.surf`.

The last two arguments are very handy in iterative training where you would want the
result of the previous iteration to be used in training the next iteration.

When you MUST train
"""""""""""""""""""""

As said earlier, the `root/Internal/slab_sim_template.blend` is the base of the slab
simulation. It has a box with width and height of 1 m and thickness of 2 mm. If,
for any reason you would need, say, a 10 mm thick slab of material, you would
have to change the thickness in the `slab_sim_template.blend` file and retrain the
nn and surf solvers.

.. note::

    If you are a developer, the work was started on making copies of the
    `slab_sim_template` so that several thickness slabs could coexist in
    the slab simulation phase, but the time run out to finish that development.
    The filename can already be passed to the blender script, but it needs
    an implementation to :py:func:`rendering.blender_control.run_slab_wl_render`
    and :py:func:`rendering.blender_control.run_parallel_slab_wl_render`.
    Another way would be to make the slab itself parametric, and just give
    the thickness as a parameter.

One last thing. It is **very important** so:

.. warning::

    There is a magical density scaling constant in
    :py:mod:`slab_simulation.slab_commons` and
    :py:mod:`blender_scripts.bs_setup_forest`. Its value is 3000. The
    particle density (as all other parameters) in slab simulation very from
    0 to 1, and this is nowhere near enough density to get the R and T values
    we want, this density scaling is used to fix it. Moreover, the value 3000
    is for 2 mm leaf slab. If you want a 1 mm thick leaf, you should raise the
    constant to 6000. For other than leaf simulation purposes, you may need
    to figure out a new good scaling factor.

    And once again, if you are a developer, please fix this ugly hard-coded
    constant that has to be synced over separate files.

That's all for slab simulation. We should be ready to get to the actual meat
of HyperBlend: :ref:`chap-system-simulation`.

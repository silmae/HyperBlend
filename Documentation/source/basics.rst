.. _chap-basics:

Beginner Tutorial
=======================

In the :doc:`previous step <install>`, we installed everything needed
to run HyperBlend, so it's time to test it out.

First Run
----------

HyperBlend does not have any kind of user interface, so you will have to
do a bit of programming to run it. If you go to the entrypoint of the
program in `root/src/__main__.py` there should only be one method which
looks something like this

.. code-block:: python3

    if __name__ == "__main__":

        runtime = initialization.initialize()

You can comment out everything there possibly is after this line and
just run the initialization to verify that everything is in order.
You can check the documentation of the initialization method from
:py:func:`setup.initialization.initialize`.

In Case of Errors
-----------------

If the Blender path is not correct or the Blender version is a wrong one
(the version is only checked on Windows), you will get an error. Check
that you installed the correct version and that you provided a correct path.

Another thing that is likely to occur is that you get an import error from
every script in the program.

The imports in HyperBlend require `<wherever_you_cloned_the_repo>/HyperBlend/`, i.e.,
the `root`, as we decided to call it in the installation tutorial,
directory to be included in `PYTHONPATH` variable.
You can check if it is by running

.. code-block:: python3

    import sys
    print(sys.path)

This will print you a list of paths in the variable and one of the should be the
`root`. If not, and if you are using PyCharm as your IDE on Windows machine, you can

- go to `<wherever_you_cloned_the_repo>/`
- right-click on HyperBlend directory
- select "Open as PyCharm project"

The IDE will then know that  `<wherever_you_cloned_the_repo>/HyperBlend/` should
be added to `PYTHONPATH`.

You can also set the `PYTHONPATH` system wide manually for any OS following
the instructions in
https://www.simplilearn.com/tutorials/python-tutorial/python-path, for example.


HyperBlend Simulations
-----------------------

This subsection contains a little introduction to HyperBlend's working
principle necessary to understand the bare basics. We will revisit these
ideas in more depth later. If you are already too eager to run some code,
you can skip to the next heading and copypaste run it, and come back here
after that.

HyperBlend is designed for creating spectral forest canopy simulations.
Since the leaves of trees are the most active part interacting with light
the first part of the simulation is focused on that. In this tutorial,
this is called with a general term :term:`Slab simulation`. Creating
custom slab simulations is explained in more detail in
:ref:`chap-slab-simulation` chapter.


The Tiny Working Example
-----------------------------

Leaf Slabs
"""""""""""""""

HyperBlend has an integrated PROSPECT :cite:`feret17` simulator for generating
leaf reflectance and transmittance spectra, so let's use that to generate a few
random leaves to work with. You can copypaste the following into the program's entry
point at `root/src/__main__.py` main method

.. code-block:: python3
    :linenos:

    from src.slab_model import interface as SMI

    if __name__ == "__main__":

        runtime = initialization.initialize()
        slab_sim_name = "tutorial_slab_simulation"
        SMI.generate_prospect_leaf_random(slab_sim_name=slab_sim_name, leaf_count=3)
        SMI.solve_slab_material_parameters(
            runtime=runtime,
            slab_sim_name=slab_sim_name,
            range_start=400,
            range_end=1000,
            resolution=100,
        )

Line ``1`` imports the slab model interface with name SMI. Line ``5`` runs the
initialization and saves the ``runtime`` environment which we will need later.
Line ``6`` is just a name of our slab simulation, which we will use the rest of this
beginner tutorial. Line ``7`` generates 3 random leaf spectra. In HyperBlend,
these pairs of reflectance and transmittance spectra are called target signal
(see :term:`Signal`). The method
:py:func:`~slab_model.interface.generate_prospect_leaf_random` doesn't have
a return value as it will save the data on the disc.

The last line runs the actual slab simulation that solves the slab (leaf) material
parameters to be used later in :term:`System simulation`. Our last call to
:py:func:`~slab_model.interface.solve_slab_material_parameters` has a lot
going on, so let's take a closer look what we are doing.

First of all, we provide it with the runtime and slab simulation name, which are
mandatory. If we did not provide any additional arguments, by default, the method
would solve us the slab material parameters with spectral range from 400 nm to
2500 nm with 1 nm spectral resolution. Which means that we would simulate 2101
spectral bands three times for all of our three leaves. To get you results a bit
faster, we restrict the range from 400 nm to 1000 nm with only 100 nm resolution,
which produces 7 spectral bands and the run should take only some seconds.
The spectral range and resolution of the slabs control the range and resolution
of the following steps, so it is a quite important step to decide beforehand, which
values to use.
You can find the results of this simulation from directory
`root/Slab simulation/tutorial_slab_simulation/`. Now we are ready to proceed to
the :term:`System simulation`.


Forest System
"""""""""""""""

In this tutorial, our system simulation is a forest scene. That is the only
one readily available by default, but it is possible to build any kind of
custom system simulations.

Copy-paste the following code snippet into the main method

.. code-block:: python3
    :linenos:

    if __name__ == "__main__":

        runtime = initialization.initialize()
        slab_sim_name = "tutorial_slab_simulation"
        system_sim_name = "tutorial_system_simulation"
        slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

        # Pack leaf data for system_simulation scene initialization.
        leaves = [
            (slab_sim_name, 0, slab_material_names[0]),
            (slab_sim_name, 1, slab_material_names[1]),
            (slab_sim_name, 2, slab_material_names[2]),
        ]

        F.init(
            leaves=leaves,
            conf_type="m2m",
            new_system_sim_name=system_sim_name,
            soil_name="median_humid_clay",
        )

        # FIRST we go this far and check the new files

Again, first, we just assign a name for the simulation and use the slab simulation
name from the previous step of the tutorial. The variable
``slab_material_names`` contains the names of the slab materials that are
present and used in the system simulation scene file, which is a Blender
(.blend) file. We'll get back to this later.

On line ``9``, we define leaves as a list of three tuples. The first element
of each tuple is the name of the slab simulation we did in the previous stage
of this tutorial. The number after it tells witch signal we want to use.
We simulated three signals in the previous step, so let's use them all.
The last element is the slab material name for which we will assign the
spectral signal to.

Then we are ready to initialize the system simulation scene. We give the
initializer :py:func:`system_simulation.forest.init` the leaves, configuration
file type master to master (more of this later), and the name for the system
simulation to be. Lastly, we decide which soil spectrum we want to use. There
are nine different precalculated soil spectra available in HyperBlend. You can
use your own too, but let's stick with the available ones for now.

When you run this, you can notice that new directory is generated:
`root/System simulation/scene_tutorial_system_simulation/`;
the name we gave for our simulation.
The most important file there is `tutorial_system_simulation.blend`, which is
our new simulation scene that is a copy of the simulation template stored in
the code repository. You can open it with Blender and take a look around if
you wish.

The other important file is called `system_sim_control.toml`, which is used to
set many parameters of the scene and is extremely useful for creating multiple
variation of a single scene, which is not covered in this beginner tutorial.
You can open it anyways, and change some values. For example, change
`hsi_resolution_x` and `hsi_resolution_y` to something small, like 64.
This will result in a small spatial size of the simulated spectral cube that
is fast to compute.

Nest step is to run the actual simulation. Comment out the `F.init` call
and add the following snippet under the previous code

.. code-block:: python3
    :lineno-start: 23

    F.setup_forest_for_rendering(
        runtime=runtime,
        system_sim_name=system_sim_name,
        slab_material_names=slab_material_names,
    )
    F.render_forest(
        runtime=runtime, system_sim_name=system_sim_name, render_mode="preview"
    )
    F.render_forest(
        runtime=runtime, system_sim_name=system_sim_name, render_mode="visibility"
    )
    F.render_forest(
        runtime=runtime, system_sim_name=system_sim_name, render_mode="spectral"
    )
    F.construct_spectral_cube(system_sim_name=system_sim_name)

This will first setup the scene. Most importantly, it assigns each frame
(HyperBlend treats frames of video as bands of spectral image) correct
material and illumination values based on the slab simulation we did earlier.

The three render calls will, first, render preview images that show the forest
scene from several perspectives. Secondly, it renders visibility maps, which tells
what material is present in each pixel (this is needed for spectral cube white
correction step soon). The last render call renders the actual bands; each as a
separate gray scale image.

The last call constructs the spectral image cube. It automatically searches
a good reference plate from the scene (using visibility maps) and performs
white correction that results in reflectance cube.

Now you have run through the simulation pipeline with a couple of dozen lines of
code and got your first spectral cube which you can open with any software that
can open ENVI-formatted spectral images. If you don't have one, you can use
HyperBlend's sister project CubeInspector that is hosted at
https://github.com/silmae/cubeinspector.

The following more advanced tutorials will show how to tailor your slab simulations
and system simulations to your needs.

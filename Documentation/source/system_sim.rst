.. _chap-system-simulation:

The System Simulation
==============================

The system simulation is quite a bit more complicated than the slab simulation
in previous tutorial. The main idea of the system simulation is still quite
simple: we take the virtual 3D slabs and put them into some bigger context
to create spectral simulations of some meaningful environment. We will
use a forest canopy as an example in this tutorial as it is the main focus
of HyperBlend, but other simulation scenarios are possible too. As an example
the already mentioned photobioreactor simulation conducted in
paper :cite:`riihiaho25`.

In the canopy simulation, we have quite a few moving parts. Firstly, the geometry,
i.e., the three dimensional shape of all of the objects has to be defined. This
includes the terrain, the trees, bushes, and so on. From spectral point of view,
we already have defined the spectral reflectance and transmittance properties for
the leaf slabs, but not for anything else. So, we will need shaders for tree trunks,
and soil. Also the illumination of the scene needs to be defined. Then there is
camera parameters like field of view (FOV), location, and orientation.

These tasks are divided between Python coding and object manipulation in Blender.
A basic understanding of at least how to navigate Blender's view and change some
simple parameters is highly recommended for the user. These skills can be acquired
by following any of the simple "donut tutorials" found on video sharing platforms
across the internet. Some differences between the navigation and view in various
Blender versions exist, so it is perhaps easiest to search for tutorials on
Blender 3.6, which is the only version supported by HyperBlend.

In the slab simulation, Blender is used under the hood of HyperBlend, and we
didn't really have to open the Blender UI at all. The system simulation is
different in that regard. The connection between HyperBlend code and Blender
is implemented through Blender's Python API. In HyperBlend, all code that calls
Blender is secluded to :py:mod:`blender_scripts` module.


How to start
--------------

If you have not read through :ref:`sec-tiny-working-example` from the
:ref:`chap-basics`, do it now. It will teach you how to create a new
forest scene with mostly default settings, so you can get up to speed immediately.
You will run a full simulation, which will result in a spectral image cube of
a forest canopy.
The tutorial at had will dig deeper into the intricacies of the process.


Initializing System Simulation
"""""""""""""""""""""""""""""""""""

In the beginner tutorial, we already had the following code that initializes
our new forest scene. We assume that there is an existing slab simulation done
with the name "tutorial_slab_simulation" (instructions in the beginner tutorial).
We will create a new system simulation with name "tutorial_system_simulation_2".

.. code-block:: python3
    :linenos:

    if __name__ == "__main__":

        runtime = initialization.initialize()
        slab_sim_name = "tutorial_slab_simulation"
        system_sim_name = "tutorial_system_simulation_2"
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
            sun_file_name="default_sun",
            sky_file_name="default_sky",
        )

Until now, it's pretty much the same as in the beginner tutorial. We are still
creating a master to master (``conf_type="m2m"``) configuration file and use the
same precalculated soil spectra as before. The new thing is that we explicitly
define the sun and the sky spectra. The names of these spectra are the default
ones, so there is no change in behaviour compared to the beginner tutorial,
but if you have other spectra available, this is how you pass them. The
sun and sky spectra files live in `root/Light spectra/`. They are simple
text files with csv formatting, so you can use any light spectra you have
at hand.

See instructions on how to generate new light spectra with supported
third party simulators from :py:mod:`system_simulation.lighting`.


Shape your ground
"""""""""""""""""

Shaping your geometry is easiest to do inside Blender's UI. Running the code
snippet will create a new system simulation and you can open the associated
Blender scene by double clicking
`root/System simulation/scene_tutorial_system_simulation_2/tutorial_system_simulation_2.blend`.

The UI should look something like this

.. image:: ../../readme_img/terrain_object_edit_parameters.png

Select the object called Ground from the to right corner and then select
the wrench icon from bottom right corner as shown.

The full list of the 39 tunable parameters is:

    - Seed
        - Random seed value
    - Size X [m]
        - Width of the forest in meters
    - Size Y [m]
        - Width of the forest in meters
    - Simplified understory
        - Replace understory objects with their convex hull for faster response time while editing
    - Minimum tree separation [m]
        - Minimum separation of objects spawn by the primary spawning mechanism
    - Spawn probability [%]
        - General spawn probability, i.e. what percentage of points of the spawn grid can be used
    - Object 1-10 probability [%] (10 separate parameters)
        - These are similar to previous, but for each spawnable object separately
    - Spawn object 1-10 (10 separate parameters)
        - The actual objects to be spawn by the tree spawning system
    - Reference object
        - Reference plate object for reflectance calculation
    - Reference controller
        - An empty object that controls the x,y position of the Reference object
    - Reference height [m]
        - Reference object's height from local ground surface
    - Reference safe distance [m]
        - Spawn objects that are closer than this to the reference object will be un-spawn
    - Height map strength
        - How strongly will the height map move the vertices of the ground on z-axis
    - Height point separation [m]
        - How detailed grid is used to apply the height map (smaller value = more detail)
    - Max tree tilt [Deg]
        - Maximum random tilt that can be assigned to spwn objects
    - Max tree scale [%]
        - Maximum random scale that can be assigned to spwn objects
    - Understory object 1-2 (2 separate parameters)
        - Objects spawned using the secondary spawning mechanism
    - Understory 1-2 min separation [m] (2 separate parameters)
        - Minimum distance between the objects spawned by the secondary spawning mechanism
    - Ground material
        - Blender material used for the soil

The list may seem daunting, but most of them are pretty simple to understand.
The main ones affecting the shape of the ground are of course the size, and
the ones related to the height map. (Using a custom image for the height
map involves a bit of tinkering inside the geometry nodes structure, so you
need to know something about Blender.)

If you want your forest to look like planted by human in neat rows, put the
minimum tree separation high, and the overall spawn probability to 100 %. If
you want to use only one model like this (every tree is the same), put the
same tree object to each ``Spawn object X``. You can create holes in your rows
by setting some of the  ``Object X probability [%]`` to less than 100.

You can call this listing (without the explanations) from
:py:func:`blender_scripts.forest_utils.list_forest_parameters`. Since the
autodoc breaks for this file, you need to check the comments from the actual
source code file.

Shape your trees
"""""""""""""""""



The Forest Control file
""""""""""""""""""""""""""

Below are the 28 adjustable parameters for a single tree model

    - Branch thickness factor (VALUE)
    - Trunk length [m] (VALUE)
    - Trunk diameter [m] (VALUE)
    - Trunk pruning (VALUE)
    - Trunk clear start [%] (INT)
    - Trunk child count (INT)
    - Trunk top spawn (BOOLEAN)
    - Branch 1 length [m] (VALUE)
    - Branch 1 diameter [m] (VALUE)
    - Branch 1 scale (VALUE)
    - Branch 1 child count (INT)
    - Branch 1 align (VALUE)
    - Droop angle [deg] (VALUE)
    - Branch 2 length [m] (VALUE)
    - Branch 2 child count (INT)
    - Branch 2 align (VALUE)
    - Branch 3 length [m] (VALUE)
    - Branch 3 align (VALUE)
    - Branch 3 resolution (INT)
    - Trunk material (MATERIAL)
    - Leaf material (MATERIAL)
    - Hide leafs (BOOLEAN)
    - Leaf object (OBJECT)
    - Leaf density (VALUE)
    - Average leaf angle (VALUE)
    - Seed (INT)
    - Splines only (BOOLEAN)
    - Hide trunk data (BOOLEAN)


Running the simulation
"""""""""""""""""""""""

hbvedfv



Lighting
----------

The lighting of the scene is completely based on an third party simulator that is
used to generate the direct sunlight and scattered skylight spectra.


Camera
----------

There is not much to do here.


Spectral materials
----------------------------

Soil
"""""""""""""""

dsafdgjrhty


Trunk
"""""""""""""""""

sadfd


Constructing the spectral cube
"""""""""""""""""""""""""""""""""

Normalization to reflectance



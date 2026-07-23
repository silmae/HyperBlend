.. _chap-system-simulation:

The System Simulation
==============================

The system simulation is quite a bit more complicated than the slab simulation
in previous tutorial. However, the main idea of the system simulation is quite
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

If you have not read through :ref:`sec-tiny-working-example` from the
:ref:`chap-basics`, do it now. It will teach you how to create a new
forest scene with mostly default settings, so you can get up to speed immediately.
You will run a full simulation, which will result in a spectral image cube of
a forest canopy.
The tutorial at hand will dig deeper into the intricacies of the process.


Initializing System Simulation
------------------------------------------

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
------------------------------------------

Shaping your geometry is easiest to do inside Blender's UI. Running the previous
code snippet will create a new system simulation and you can open the associated
Blender scene by double clicking
`root/System simulation/scene_tutorial_system_simulation_2/tutorial_system_simulation_2.blend`.

The UI should look something like this

.. image:: ../../readme_img/terrain_object_edit_parameters.png

Select the object called Ground from the top right corner and then select
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

        - Maximum random tilt that can be assigned to spawn objects

    - Max tree scale [%]

        - Maximum random scale that can be assigned to spawn objects

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
Sphinx autodoc breaks for this file, the above link does not work and you
need to check the comments from the actual source code file.

Shape your trees
------------------------------------------

The trees are also parametric models. To adjust them to your liking,
you should first hide the ground object, so that you can see what you
are doing. This can be done by clicking the eye icon on the top right
corner from the row with the `Ground` object (can be seen in the previous
picture).

The picture below shows how to unhide the `Trees` object collection
and any of the trees you want to reshape by clicking the eye icons.

.. image:: ../../readme_img/adjust_tree_params.png

Then you can adjust the parameters to your liking as before under
the wrench icon.
Below are the 28 adjustable parameters for the tree model
(obvious explanations omitted)

    - Branch thickness factor (VALUE)
    - Trunk length [m] (VALUE)
    - Trunk diameter [m] (VALUE)
    - Trunk pruning (VALUE)

        - Removes some of the branch spawning points to make airier crown

    - Trunk clear start [%] (INT)

        - Percentage of the trunk length that is left completely branchless

    - Trunk child count (INT)

        - How many first-level branches will be spawn on the trunk

    - Trunk top spawn (BOOLEAN)

        - If True, spawn one branch as a continuation of the trunk

    - Branch 1 length [m] (VALUE)
    - Branch 1 diameter [m] (VALUE)

    - Branch 1 scale (VALUE)

        - Scale 1st level branches bigger or smaller when approaching the top of the trunk

    - Branch 1 child count (INT)

        - How many 2nd level branches are spawn to the 1st level branch

    - Branch 1 align (VALUE)

        - Alignment of the 1st level branches to the tree trunk

    - Droop angle [deg] (VALUE)

        - How curved the branches will be

    - Branch 2 length [m] (VALUE)
    - Branch 2 child count (INT)
    - Branch 2 align (VALUE)
    - Branch 3 length [m] (VALUE)
    - Branch 3 align (VALUE)

    - Branch 3 resolution (INT)

        - How many faces will the last level branch have. This affects to how many leaves are spawn

    - Trunk material (MATERIAL)
    - Leaf material (MATERIAL)
    - Hide leafs (BOOLEAN)

    - Leaf object (OBJECT)

        - Select a leaf object to be used

    - Leaf density (VALUE)

        - More or less leaves

    - Average leaf angle (VALUE)

    - Seed (INT)

        - Random seed for this tree

    - Splines only (BOOLEAN)

        - Remove actual geometry and show only splines constructing the tree. For debugging

    - Hide trunk data (BOOLEAN)

        Trees can show some data of themselves. Must be hidden during rendering



Lighting
----------

The lighting of the scene is completely based on an third party simulator that is
used to generate the direct sunlight and scattered skylight spectra. By default,
there are two light spectra available in `root/Light spectra` called `default_sky.txt`
and `default_sun.txt`. The content of the default sun file begins like this

.. code-block::

    # Output file from SSolar_GOA model
    #   Input Data
    #  Ext. Spectrum= Wehrli
    # lat=41.66 lon=-4.7 sza=29.358 jday=152
    #  p=1013.0 o3=300.0 h2o=1.5 alb=0.2
    #  alpha=1.5 beta=0.05 Wa=0.98 g=0.75
    # Wavelength Irradiance
    400.0 0.77937
    401.0 0.84119
    402.0 0.87123
    403.0 0.87947
    404.0 0.87883
    ...

It is a simple CSV-formatted file with wavelengths and irradiances. Rows
prefixed with `#` are not read and can be used for comments and metadata.
The light files are directly read during the forest setup, where you can
also provide the name of the file(s) to be used.


Camera
----------

For top of canopy imaging setup, it is easiest to control the camera using
the control file. However, you may want to test for a good combination
of altitude and FOV for your forest scene (depending on scene dimensions)
through Blender UI.


The control file
-----------------------------

**The control file** is used to store values that define the geometry
of the scene. In other words it has the same values that were listed
above and many more. An extract of a control file is shown below

.. code-block::

    Note = "This file controls the setup of the Blender scene file. "
    is_master_control = true

    [Sun]
    Note = "When sun azimuth angle is 0 degrees, the sun points to positive y-axis direction in Blender that is thought as north in HyperBlend. 90 degrees would be pointing west, 180 to south and 270 to east, respectively. Zenith angle is the Sun's angle from zenith."
    sun_angle_zenith_deg = 23.550000484583723
    sun_angle_azimuth_deg = 335.91998685373596
    sun_base_power_hsi = 4
    sun_base_power_rgb = 100

    [Drone]
    Note = "Unit of drone location and altitude is meter."
    drone_location_x = 0.0
    drone_location_y = 0.0
    drone_altitude = 100.0

    [Cameras]
    Note = "Camera angles are stored in degrees in this file. They must be converted to radians before passing to Blender file."
    drone_hsi_fow = 28.000001917535847
    drone_rgb_fow = 28.000001917535847

    [Rendering]
    Note = "Sample count controls how many samples (light rays) are cast through each pixel.More samples result in smoother image but require more time to render. Try values between 16 and 512, for example. The RGB sampling is for preview images so it can be higher as not many images are rendered with that sampling."
    sample_count_rbg = 32
    sample_count_hsi = 32

    [Images]
    hsi_resolution_x = 512
    hsi_resolution_y = 512
    rgb_resolution_x = 512
    rgb_resolution_y = 512
    walker_resolution_x = 512
    walker_resolution_y = 512
    sleeper_resolution_x = 512
    sleeper_resolution_y = 512
    tree_preview_resolution_x = 512
    tree_preview_resolution_y = 512

    [Forest.Seed]
    Value = 4
    "Standard deviation" = 1
    Type = "INT"
    ID = 7

    [Forest."Size X [m]"]
    Value = 75.0
    Type = "VALUE"
    ID = 8

    [Forest."Size Y [m]"]
    Value = 75.0
    Type = "VALUE"
    ID = 9

    ...
    # Here are more terrain object parameters
    # These comment is not part of the file format
    # Then there are all the tree parameters
    ...

    [Forest."Spawn object 1"."Trunk length [m]"]
    Value = 15.0
    "Standard deviation" = 1.5
    Type = "VALUE"
    ID = 10

    [Forest."Spawn object 1"."Trunk diameter [m]"]
    Value = 0.4000000059604645
    "Standard deviation" = 0.04000000059604645
    Type = "VALUE"
    ID = 11

    [Forest."Spawn object 1"."Trunk pruning"]
    Value = 0.5999999046325684
    "Standard deviation" = 0.05999999046325684
    Type = "VALUE"
    ID = 12

    [Forest."Spawn object 1"."Trunk clear start [%]"]
    Value = 30
    "Standard deviation" = 3
    Type = "INT"
    ID = 13
    ...

And so on. The control file can be several hundreds lines long depending on
how many spawn objects are used in the scene. Most of it is meant to be edited
programmatically The file is toml-formatted human readable text, which is read
into a Python dictionary.

Usage
""""""""""

The control file is coupled with the Blender scene file, as already shown in
the :ref:`chap-basics`. **Remember** that you have to take care to bring any
manually made changes from the scene to the control file and from the control
file to the scene file! In other words, they are cyclically dependent of each
other. Syncing the files is done through
:py:func:`system_simulation.forest.process_forest_control`, so run it
after any changes. When either of the files is modified through code, they
are synced automatically.

Manual changes to the control file usually relate to the general parameters
at the beginning of the file, i.e., light power, spatial image size,
sample count, etc. while you are searching for good settings for your
specific scene.

The real power of the control file is to create randomized variations of
your scene.
Variables with field ``"Standard deviation"`` are randomized when you initialize a
new "m2s" system simulation with :py:func:`system_simulation.forest.init`.
In other words, you would start with an ancestor,
or a *master*, system sim. The master's control file will have these fields
that are read, and new values for the *slave* system sim are populated by
drawing from gaussian distribution with the original value as mean and
standard deviation as variance. Only fields with ``Type="INT"`` or
``Type="VALUE"`` (float) are randomized. Seeds have a special randomization
where a value is drawn from uniform distribution between 0 and 10000.
The ``ID`` field is the variable's id inside Blender scene.

.. note::
    The slave control files do not have the ``"Standard deviation"`` field.


Changing the global master control file
"""""""""""""""""""""""""""""""""""""""""""""""""""

.. warning::
    This action is for developers only.

If you want to change the global master control file, i.e., the one shipped
in the repository, you can do it like this after making changes to the
forest scene template in `root/Internal`

.. code-block:: python3

    from src.rendering import blender_control as BC

    if __name__ == "__main__":

        runtime = initialization.initialize()
        rng = np.random.default_rng(1234)

        # REWRITES GLOBAL MASTER !!!!!!!!!!!!!!!!!
        BC.process_forest_control(
            runtime=runtime,
            global_master=True,
            generate=True,
        )

In reverse, if you make changes to the control file and want to apply it to the template
scene, simply set ``generate=False``. See API documentation
:py:func:`rendering.blender_control.process_forest_control`.




Spectral materials
----------------------------

Soil
"""""""""""""""

Generating a new soil spectrum with the integrated soil spectrum
generator GSV is simple:

.. code-block:: python3

    from src.gsv import interface as gsvi

    if __name__ == "__main__":

        runtime = initialization.initialize()
        soil_spectra = gsvi.simulate_gsv_soil(c1=0.528, c2=-0.011, c3=0.014, cSM=-0.129)
        gsvi.write_soil_spectra(reflectance_spectra=soil_spectra, filename="My new soil spectrum")

The example given here recreates the humid clay spectrum already included in
the default soil spectra in the repository. Consult the original GSV paper
:cite:`gsv19` and API documentation :py:mod:`gsv.interface` for the explanation
of the parameters.

Running the above snippet will add a new csv file to `root/Reflectance spectra`.

It begins like this:

.. code-block::

    400.0 0.223112120
    401.0 0.223672999
    402.0 0.224233878
    403.0 0.224794757
    404.0 0.225355636
    405.0 0.225916515
    406.0 0.226477394
    407.0 0.227038273

The default soil spectra included in the repository look like this

.. image:: ../../readme_img/default_soils.png


Trunk
"""""""""""""""""

There is no integrated simulator for trunk data in HyperBlend largely because
such a simulator could not be found from literature. Someone should build one.
That someone could be you!

Anyways, there was no time to implement proper spectral trunks even though
it should not be very difficult. At the moment, trunk shader is set to be
0.25 reflective diffuse material in
:py:func:`blender_scripts.bs_setup_forest.insert_diffuse_material_spectra`.
There are some comments to help you implement it if you are up for it.


Constructing the spectral cube
"""""""""""""""""""""""""""""""""

Constructing a spectral image cube from your simulation is very simple

.. code-block:: python3

    from src.system_simulation import forest as F

    if __name__ == "__main__":

        runtime = initialization.initialize()

        system_sim_name = "My simulation"
        F.construct_spectral_cube(system_sim_name=system_sim_name)

Before calling that, you should have rendered your spectral bands and
visibility maps for the scene. It will automatically find the reference
plates for reflectance calculation and select the brightest one that does
not cause over-exposure. If all of them do, you should lower your light
power in the control file and rerender all bands.

See full instructions of the method
:py:func:`system_simulation.forest.construct_spectral_cube`.

You can find the spectral cube from
`root/System simulation/scene_My simulation/Spectral cube`.
The cube is saved in ENVI format, so there are two files:
.hdr and .img file. The .hdr file is just headers and
looks something like this

.. code-block::

    ENVI
    samples = 1024
    lines = 1024
    bands = 421
    header offset = 0
    file type = ENVI Standard
    data type = 4
    interleave = bip
    byte order = 0
    data_type = 4
    default bands = { 47 , 27 , 14 }
    wavelength = { 400.0 , 405.0 , 410.0 , 415.0 , 420.0 , 425.0 , 430.0 , 435.0 , 440.0 , 445.0 , 450.0 , 455.0 , 460.0 , 465.0 , 470.0 , 475.0 , 480.0 , 485.0 , 490.0 , 495.0 , 500.0 , 505.0 , 510.0 , 515.0 , 520.0 , 525.0 , 530.0 , 535.0 , 540.0 , 545.0 , 550.0 , 555.0 , 560.0 , 565.0 , 570.0 , 575.0 , 580.0 , 585.0 , 590.0 , 595.0 , 600.0 , 605.0 , 610.0 , 615.0 , 620.0 , 625.0 , 630.0 , 635.0 , 640.0 , 645.0 , 650.0 , 655.0 , 660.0 , 665.0 , 670.0 , 675.0 , 680.0 , 685.0 , 690.0 , 695.0 , 700.0 , 705.0 , 710.0 , 715.0 , 720.0 , 725.0 , 730.0 , 735.0 , 740.0 , 745.0 , 750.0 , 755.0 , 760.0 , 765.0 , 770.0 , 775.0 , 780.0 , 785.0 , 790.0 , 795.0 , 800.0 , 805.0 , 810.0 , 815.0 , 820.0 , 825.0 , 830.0 , 835.0 , 840.0 , 845.0 , 850.0 , 855.0 , 860.0 , 865.0 , 870.0 , 875.0 , 880.0 , 885.0 , 890.0 , 895.0 , 900.0 , 905.0 , 910.0 , 915.0 , 920.0 , 925.0 , 930.0 , 935.0 , 940.0 , 945.0 , 950.0 , 955.0 , 960.0 , 965.0 , 970.0 , 975.0 , 980.0 , 985.0 , 990.0 , 995.0 , 1000.0 , 1005.0 , 1010.0 , 1015.0 , 1020.0 , 1025.0 , 1030.0 , 1035.0 , 1040.0 , 1045.0 , 1050.0 , 1055.0 , 1060.0 , 1065.0 , 1070.0 , 1075.0 , 1080.0 , 1085.0 , 1090.0 , 1095.0 , 1100.0 , 1105.0 , 1110.0 , 1115.0 , 1120.0 , 1125.0 , 1130.0 , 1135.0 , 1140.0 , 1145.0 , 1150.0 , 1155.0 , 1160.0 , 1165.0 , 1170.0 , 1175.0 , 1180.0 , 1185.0 , 1190.0 , 1195.0 , 1200.0 , 1205.0 , 1210.0 , 1215.0 , 1220.0 , 1225.0 , 1230.0 , 1235.0 , 1240.0 , 1245.0 , 1250.0 , 1255.0 , 1260.0 , 1265.0 , 1270.0 , 1275.0 , 1280.0 , 1285.0 , 1290.0 , 1295.0 , 1300.0 , 1305.0 , 1310.0 , 1315.0 , 1320.0 , 1325.0 , 1330.0 , 1335.0 , 1340.0 , 1345.0 , 1350.0 , 1355.0 , 1360.0 , 1365.0 , 1370.0 , 1375.0 , 1380.0 , 1385.0 , 1390.0 , 1395.0 , 1400.0 , 1405.0 , 1410.0 , 1415.0 , 1420.0 , 1425.0 , 1430.0 , 1435.0 , 1440.0 , 1445.0 , 1450.0 , 1455.0 , 1460.0 , 1465.0 , 1470.0 , 1475.0 , 1480.0 , 1485.0 , 1490.0 , 1495.0 , 1500.0 , 1505.0 , 1510.0 , 1515.0 , 1520.0 , 1525.0 , 1530.0 , 1535.0 , 1540.0 , 1545.0 , 1550.0 , 1555.0 , 1560.0 , 1565.0 , 1570.0 , 1575.0 , 1580.0 , 1585.0 , 1590.0 , 1595.0 , 1600.0 , 1605.0 , 1610.0 , 1615.0 , 1620.0 , 1625.0 , 1630.0 , 1635.0 , 1640.0 , 1645.0 , 1650.0 , 1655.0 , 1660.0 , 1665.0 , 1670.0 , 1675.0 , 1680.0 , 1685.0 , 1690.0 , 1695.0 , 1700.0 , 1705.0 , 1710.0 , 1715.0 , 1720.0 , 1725.0 , 1730.0 , 1735.0 , 1740.0 , 1745.0 , 1750.0 , 1755.0 , 1760.0 , 1765.0 , 1770.0 , 1775.0 , 1780.0 , 1785.0 , 1790.0 , 1795.0 , 1800.0 , 1805.0 , 1810.0 , 1815.0 , 1820.0 , 1825.0 , 1830.0 , 1835.0 , 1840.0 , 1845.0 , 1850.0 , 1855.0 , 1860.0 , 1865.0 , 1870.0 , 1875.0 , 1880.0 , 1885.0 , 1890.0 , 1895.0 , 1900.0 , 1905.0 , 1910.0 , 1915.0 , 1920.0 , 1925.0 , 1930.0 , 1935.0 , 1940.0 , 1945.0 , 1950.0 , 1955.0 , 1960.0 , 1965.0 , 1970.0 , 1975.0 , 1980.0 , 1985.0 , 1990.0 , 1995.0 , 2000.0 , 2005.0 , 2010.0 , 2015.0 , 2020.0 , 2025.0 , 2030.0 , 2035.0 , 2040.0 , 2045.0 , 2050.0 , 2055.0 , 2060.0 , 2065.0 , 2070.0 , 2075.0 , 2080.0 , 2085.0 , 2090.0 , 2095.0 , 2100.0 , 2105.0 , 2110.0 , 2115.0 , 2120.0 , 2125.0 , 2130.0 , 2135.0 , 2140.0 , 2145.0 , 2150.0 , 2155.0 , 2160.0 , 2165.0 , 2170.0 , 2175.0 , 2180.0 , 2185.0 , 2190.0 , 2195.0 , 2200.0 , 2205.0 , 2210.0 , 2215.0 , 2220.0 , 2225.0 , 2230.0 , 2235.0 , 2240.0 , 2245.0 , 2250.0 , 2255.0 , 2260.0 , 2265.0 , 2270.0 , 2275.0 , 2280.0 , 2285.0 , 2290.0 , 2295.0 , 2300.0 , 2305.0 , 2310.0 , 2315.0 , 2320.0 , 2325.0 , 2330.0 , 2335.0 , 2340.0 , 2345.0 , 2350.0 , 2355.0 , 2360.0 , 2365.0 , 2370.0 , 2375.0 , 2380.0 , 2385.0 , 2390.0 , 2395.0 , 2400.0 , 2405.0 , 2410.0 , 2415.0 , 2420.0 , 2425.0 , 2430.0 , 2435.0 , 2440.0 , 2445.0 , 2450.0 , 2455.0 , 2460.0 , 2465.0 , 2470.0 , 2475.0 , 2480.0 , 2485.0 , 2490.0 , 2495.0 , 2500.0 }
    wavelength units = nm

The .img file contains the actual data. You can open the files programmaticly
with Python package spectral like so

.. code-block:: python3

    from spectral.io import envi as envi

    cube_data = envi.open(file=hdr_path, image=data_path)

For making it easier, you can just download an executable from
HyperBlend's sister project CubeInspector https://github.com/silmae/cubeinspector


Bundle system simulation
--------------------------------

The bundle run is the holy grail of HyperBlend. This is where you
can truly generate arbitrary number of variations of a system simulation
scene by just two lines of code. In the example below, we assume we have
done slab simulation with name `"slabs_for_bundle"` and then create a
ancestor scene. Finally, the ancestor is used to generate three new
variants, and directly processed into spectral image cubes.

.. code-block:: python3

    if __name__ == "__main__":

        runtime = initialization.initialize()

        rng = np.random.default_rng(666)

        slab_sim_name = "slabs_for_bundle"
        slab_material_names = ["Slab material 1", "Slab material 2", "Slab material 3"]

        # Pack leaf data for system_simulation scene initialization.
        leaves = [
            (slab_sim_name, 0, slab_material_names[0]),
            (slab_sim_name, 1, slab_material_names[1]),
            (slab_sim_name, 3, slab_material_names[2]),
        ]

        bundle_ancestor_name = "bundle ancestor"
        F.init(
            leaves=leaves,
            conf_type="m2m",
            new_system_sim_name=bundle_ancestor_name,
        )

        bundle_name = "My second bundle"
        F.create_scene_bundle(bundle_name=bundle_name, system_sim_name_ancestor=bundle_ancestor_name, rng=rng, count=3, leaves=leaves)
        F.run_scene_bundle(runtime=runtime, bundle_name=bundle_name, slab_material_names=slab_material_names)


You will find these new system simulations from `root/System simulation/`
like any other system simulation. The names of the directories are generated
with a time stamp that is unique for all.

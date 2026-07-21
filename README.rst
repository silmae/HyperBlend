HyperBlend
=========================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


Mini info:

    - Access the full documentation and tutorials at https://hyperblend.readthedocs.io/en/latest/index.html
    - Supported operating systems: Windows and Linux (tested on Windows 10 and Ubuntu 20.0.4).
    - Programming language: Python.
    - Active development from May 2021 to September 2026. Development may resume later in this or another fork.

HyperBlend is a spectral simulator that can be used to construct complex scenes with
physically based light propagation. Its main focus is in simulating forest canopies
but it has also been utilized in other applications, such as modeling illumination
conditions inside photobioreactors :cite:`riihiaho25`. HyperBlend's most distinct
benefit over many other spectral simulators are that it can be applied to various
simulation needs in different fields and its capability to represent virtually any
object with arbitrary level of detail.

HyperBlend simulation is performed in two distinct phases: the slab simulation phase
and the system simulation phase. These rather generic names highlight
HyperBlend's generalizability to various simulation needs.

The leaf simulation depicts a leaf as a flat box shape, which is filled with
scattering and absorbing particles. Desired spectral reflectance and transmittance
properties are found by adjusting the density and proportions of these particles
as well as their forward and backward scattering tendencies. In essence, the leaf
simulation needs a target reflectance and transmittance as a target spectra, which
may originate from real-world leaf measurements or from a leaf optical properties
model, such as PROSPECT :cite:`jacquemoud90`. The target spectra are then reproduced
in the virtual environment to be incorporated into the canopy simulation later. The
details of the leaf simulation phase are discussed in :cite:`riihiaho22,riihiaho23`.

Once the leaf objects have their proper spectral properties, they can be placed
in trees, and the trees onto ground to form a forest. The main idea in HyperBlend's
canopy simulation is the capability to produce variations of forests with
minimal manual effort. This is achieved by utilizing procedural 3D-modeling approach
that allows the user to define a desired base geometry and then let the automatic
randomization system to generate more geometrical variants.

HyperBlend supports procedural modeling in construction of individual objects and
full scenes through Blender :cite:`blender`. In traditional 3D modeling, a 3D artist would adjust
the position of vertices (sing. vertex) to achieve a desired shape of a solid or a
surface. In procedural 3D modeling, the shape of the objects is defined using noise
maps, curves, and primitive geometric shapes such as cubes. These primitive shapes
are scaled, rotated, deformed, and connected to each other to crate more complex
shapes. Variation and randomness occurring in nature is driven by noise maps.
One can think of them as random number generators in 3D-world. HyperBlend also supports
the traditional manual 3D-model creation and even third-party models can be imported
to the system with relative ease.

The tree objects spawn on the terrain vary in shape and size. Geometrically, a tree
in HyperBlend is one vertical general cylinder with branching general cylinders glued
to its sides. The number of individual tree objects in a scene varies from one to ten.
The objects spawn on the terrain are instanced from these original tree objects with
random rotation and scaling. Each scene can contain dozens or even hundreds of tree
instances. All original tree objects utilize a single procedural tree model with
different parameters. In other words, for creating a new tree geometry, one can simply
copy an existing tree, change its parameters, and a completely different looking
tree is created. The tree model has almost thirty parameters exposed for the user.
The set of exposed parameters contain the most important ones for affecting the
appearance of the tree while still keeping the number of parameters low enough not
to be exhausting for the user. These parameters include: trunk and branch lengths and
thicknesses, controls for where and how many branches are created, branch alignment and
curvature, leaf density and average leaf angle.
Of course, proficient Blender users can access many more parameters inside the model
but only the exposed ones are automatically randomizable without changes to the source
code.

The terrain object is a plane and its main purpose is to spawn tree objects to form
a forest. It is also a procedural model and its own shape is controlled by a few parameters,
mainly the size and the undulation of the plane. The undulation is controlled by a
simple gray-scale image, so if desired, one can use an actual height map for creating
a shape similar to, e.g., a real-world forest site. By default, undulation is controlled
by a noise texture. The terrain object is also responsible for spawning a reference
control panel that is used to calculate a reflectance image. The reference panel's
position is adjustable, and it will de-spawn all trees from around it (at adjustable
distance) to avoid it being shadowed, or not seen at all, under the canopy.

The terrain object consist of two kinds of spawning systems: one for the trees, and
one for the understory. The tree spawning system ensures a minimum separation for
each tree. It can be used to generate highly heterogeneous forests depicting natural
forest sites or, with different parameters, it can represent a homogenous, planted
trees in neat rows. It's also possible to create scenes with something in-between.
For example, having neat rows of trees, but some slots are missing a tree.

The second spawning system for understory also enforces a minimum distance for
the objects spawned by this system, but it does not take into account any other
objects in the scene. In other words, it can spawn objects at the same spot that
a tree is already occupying. This system is in place for creating an undisturbed
cover of bushes or whatever elements one wants to place to the scene. The reference
panel does not affect this system in any way. It is assumed (and can be enforced)
that the reference panel is higher than the understory. Together, these spawning systems
allow for a natural two layered canopy structure to be created. If more canopy layers
are needed, one can use the tree height to achieve that, though the trees will never
spawn directly under each other.

Currently, HyperBlend supports ten combinations of tree geometry and leaf spectra per
scene. In other words, one can use, for example, two geometrical trees, both with five
spectral variations or five geometrical variants with two different spectra at maximum.
The leaf spectrum for each tree geometry is the same in all the leaves of that tree.
Both, instantiating the trees from a few ancestors and using
only a single leaf spectra per tree allows for very fast simulation times even in highly
complex overall scene geometry.

In spectral sense, the terrain has a homogenous, diffuse, spectral soil shader.
The soil spectra in HyperBlend is powered by GSV soil spectra generator :cite:`gsv19`.
There are nine precalculated soil spectra in HyperBlend's code repository ready for
use. They are derived from the work in :cite:`gsv19`. They are combinations of three
humidity levels (wet, humid, dry) and three soil materials (sand, peat, clay).
The benefit of GSV is, that it can take a multispectral soil measurement as an
input, and extend it to a hyperspectral spectrum. One should note that the tree
trunk shaders are not spectral, but a constant (defined by user) diffuse reflecting
surface on all spectral bands.

The canopy simulation has two light sources: direct sunlight and scattered skylight.
The angular size of the sun is physically accurately set at :math:`0.545^{\circ}`.
This ensures physically accurate shadow edges. The illuminating power
of the sky scattered light is constant over the half hemisphere and is thus not
physically accurate. The user is free to provide the spectra for both light sources
in a comma separated value (CSV) formatted text file. Both light spectra are automatically
normalized so that both spectra are divided by the brightest band in the sun spectrum.
This places a restriction that the sunlight must be brighter than the sky, i.e.,
it is not possible to simulate strongly overcast weather. When rendering the final
band frames with Blender, the light spectra are scaled with appropriate (user-defined)
value to mimic exposure time. Thus, the pixel values should not be interpreted as
physically meaningful irradiance values, and the resulting spectral image cube can
only be used as a reflectance image.

The virtual spectral camera in the scene is located at the world origin at height defined
by the user. There are other cameras to render RGB preview images of the forest but they
are of less importance, we can talk about the spectral camera as
*the* camera. The camera has adjustable field of view (FOV), to account for
different imaging heights and scene dimensions. The spatial resolution of the camera
is also adjustable.
The camera projection is perspective, which will not perfectly mimic push-broom
type imagers, which are perspective in across-track direction and orthographic in
along-track direction. Apart from rendering noise originating from the Monte Carlo
process in Blender's Cycles rendering engine, the resulting band frames should be
considered "perfect". In other words, no aberrations that would occur in
real-world images, such as upwelling atmospheric effects or lens aberrations,
are present in the images. The band frames thus resemble real-world images after
being perfectly corrected. One should note that the down-welling atmospheric effects
should be included in the light spectra used in the simulation.

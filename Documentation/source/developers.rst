.. _chap-developers:

For Developers
=======================

Some notes for possible future developers of HyperBlend.

First of all, the refactoring has not been fully done in 2026, so
some old variable names will probably exist deeper in the code. That's
why they are still included in the :ref:`chap-glossary`. Similarly, the quality
of documentation will probably decline the deeper (or less frequently used
parts) into the code you travel. The most effort in documentation and
refactoring was put close to the front of the code, especially the interfaces.


Directory names
------------------

If you add directories or rename old ones, you should update the names
in :py:mod:`src.constants`, `root/src/definitions/directory_structure.toml`,
and :py:mod:`data.path_handling`. They all work in tandem to keep the directory
structure in order and automatically create directories that are not there.

Ideas for future development
-------------------------------

Support for spectral trunks is non-existent. Should be easy to implement
using the same idea as with soil spectra.

The light normalization should be reworked so that the sky scattering
can be brighter than the direct sunlight. This would allow simulating
strongly overcast skies. Taking that further: would be cool to have
clouds with shadows too, but that is a very challenging problem to
get physically accurate.

It would be more realistic the leaves to have a surface. This would
require building a virtual goniometric lab in the same manner as
the slab simulation is currently built. At least LOTUS :cite:`lotus24`
dataset has goniometric measurements available, so this could be used to
build accurate shader.

.. HyperBlend documentation master file, created by
   sphinx-quickstart on Tue Apr 15 10:03:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HyperBlend's documentation!
======================================

This documentation is still work in progress.

If this is the first time you hear about HyperBlend, you might want
to read this page first to gain some information on what kind of simulator
it is. You can then continue to :ref:`chap-install` for installation instructions
and then to :ref:`chap-basics` for a beginner tutorial to construct your first HyperBlend
simulation.

This documentation lives in https://hyperblend.readthedocs.io/en/latest/index.html
The project is hosted in GitHub at https://github.com/silmae/HyperBlend
The latest simulated dataset for validation is available in Zenodo with DOI: https://doi.org/10.5281/zenodo.17140892

All usage examples are in the tutorials, there are none in the API documentation.
The documentation is organized as follows:

.. The toctree defines the structure of the documentation on the top level.
   The actual toc is not shown at this location as it is included in the sidebar.
.. toctree::
   :glob:
   :maxdepth: 2
   :name: mastertoc
   :caption: Contents:

   self
   install
   basics
   slab_sim
   system_sim
   developers
   code/src
   license
   contributing
   glossary
   references

Readme
=========

This is the same README file that is shown on the github landing page.

.. The README file is included here to provide a quick overview of the project.
   It will be displayed on the main page of the documentation. We have to parse it
   with myst_parser to include it correctly as it is written in markdown.
   :parser: myst_parser.sphinx_
.. include:: ../../README.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. HyperBlend documentation master file, created by
   sphinx-quickstart on Tue Apr 15 10:03:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HyperBlend's documentation!
======================================

This documentation is still work in progress. We will get it done eventually!

The documentation is organized as follows:

.. The toctree defines the structure of the documentation on the top level.
   The actual toc is not shown at this location as it is included in the sidebar.
.. toctree::
   :maxdepth: 2
   :name: mastertoc
   :caption: Contents:

   self
   install
   slab_sim
   system_sim
   src
   license
   contributing

Readme
=========

This is the same README file that is shown on the github landing page.

.. The README file is included here to provide a quick overview of the project.
   It will be displayed on the main page of the documentation. We have to parse it
   with myst_parser to include it correctly as it is written in markdown.
.. include:: ../../README.md
   :parser: myst_parser.sphinx_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



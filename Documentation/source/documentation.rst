.. _chap-documentation:

Documentation
==============================

HyperBlend documentation, the one you are reading now, is generated with Sphinx.

Use these settings to generate the docs:

    - Command: `html`
    - Input directory: `root/Documentation/source/`
    - Output directory: `root/Documentation/build/`

        - If the directory does not exist, just create it
        - All html files are generated here. You can access the docs locally
          by double clicking the index.html file
    - Use the same python interpreter as for running the code normally
    - No other options are needed

Readthedocs
----------------------------

The documentation is hosted at
https://hyperblend.readthedocs.io/en/latest/index.html
The build of the documentation is done automatically on the server
when a new push is made to the GitHub repository.

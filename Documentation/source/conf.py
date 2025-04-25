# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# #
# # from matplotlib import pyplot as plt
# #
# # from PIL import Image
for x in os.walk('../../src'):
  sys.path.insert(0, os.path.abspath(x[0]))



# sys.path.insert(0, os.path.abspath("../../src/gsv/"))
# print(sys.path)

project = 'HyperBlend'
copyright = '2025, Kimmo Riihiaho'
author = 'Kimmo Riihiaho'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','myst_parser']

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True
}

autodoc_mock_imports = ['bpy',]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Imports needed for autodoc ----------------------------------

import os
import sys

os.chdir("../../")

#sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("."))

#sys.path.append(os.path.abspath("./src/blender_scripts/"))

#for x in os.walk("../../src"):
    #sys.path.insert(1, os.path.abspath(x[0]))
    #sys.path.append(os.path.abspath(x[0]))

for x in os.walk("./src"):
    sys.path.insert(1, os.path.abspath(x[0]))

# -- Project info --------------------------------------------

project = "HyperBlend"
copyright = "2025, Kimmo Riihiaho"
author = "Kimmo Riihiaho"
release = "0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    #    "myst_parser",
    "sphinx.ext.mathjax",  # for Latex math
    # "sphinxcontrib.bibtex",  # bibtex style citations
]

bibtex_bibfiles = ["references.bib"]  # tells bibtex which file to use

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}

add_module_names = False  # clear module names out of function signatures

autodoc_mock_imports = [
    "bpy",  # Do not try to truly import Blender's bpy
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

import sys
import os
sys.path.insert(0, os.path.abspath('../src/')) #pts to outside doc
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'catapult'
copyright = '2025, Hayden Prairie, Frank Collebrusco, Arsh Guntakal, Siddharth Benoy'
author = 'Hayden Prairie, Frank Collebrusco, Arsh Guntakal, Siddharth Benoy'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# Napoleon Settings
napoleon_google_docstring = True  # Enable Google-style docstrings
#napoleon_numpy_docstring = False  # Disable NumPy-style docstrings (optional)
napoleon_include_init_with_doc = True  # Include __init__ docstring in class docs
#napoleon_use_param = True  # Use :param: for function parameters
#napoleon_use_rtype = True  # Use :rtype: for return types



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

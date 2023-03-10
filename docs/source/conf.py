# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pdb_numpy'
copyright = '2023, Samuel Murail'
author = 'Samuel Murail'
release = '0.0.1'

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = [
#    'sphinx.ext.autodoc',
#    'sphinx.ext.todo',
#    'sphinx.ext.githubpages',
#    'sphinxarg.ext',
#    'sphinx.ext.mathjax',
#    'nbsphinx',
#]

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'numpydoc']

templates_path = ['_templates']
exclude_patterns = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.ipynb']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme ='sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
# The master toctree document.
master_doc = 'index'

man_pages = [
    (master_doc, 'pdb_numpy',
     'PDB Numpy Documentation',
     [author], 1)
]

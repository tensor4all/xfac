import subprocess, os

# Check if we're running on Read the Docs' servers
#read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}

#if read_the_docs_build:
subprocess.call('doxygen', shell=True)
breathe_projects['xfac'] = 'doxygen_out/xml'

# -- Project information -----------------------------------------------------

project = 'xfac'
copyright = '2024, Tensor4all'
author = 'Tensor4all team'

from cgitb import html

extensions = ["breathe"]

html_theme = "sphinx_rtd_theme"

# Breathe configuration
breathe_default_project = "xfac"

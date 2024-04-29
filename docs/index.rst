Welcome to xfac's documentation!
================================

.. toctree::
  :maxdepth: 2
  :caption: Contents:

:ref:`genindex`

Installation
============

To install xfac you just have to follow the instructions at the README.md file that you can get from the repo:
https://github.com/tensor4all/xfac

Factorize a (discrete) tensor function
======================================

.. doxygenstruct:: xfac::TensorCI2
    :members:

Factorize a continuous function using natural grid
==================================================

.. doxygenfunction:: xfac::grid::QuadratureGK15

.. doxygenclass:: xfac::CTensorCI2
    :members:

Factorize a continuous function using quantics grid
===================================================


.. doxygenstruct:: xfac::grid::Quantics
    :members:

.. doxygenclass:: xfac::QTensorCI
    :members:



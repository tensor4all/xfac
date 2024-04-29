Welcome to xfac's documentation!
================================

This is the c++ API documentation.

There are some `python tutorials`_.

.. _`python tutorials`: https://github.com/tensor4all/xfac/tree/main/docs/tutorial-python


Contents:
---------

- `Installation`_
- `Factorize a (discrete) tensor function`_
- `Factorize a continuous function using natural grid`_
- `Factorize a continuous function using quantics grid`_
- `Automatic building of matrix product operators (autoMPO)`_

.. toctree::
    :maxdepth: 2


Installation
============

To install xfac you just have to follow the instructions at the README.md file that you can get from the repo:
https://github.com/tensor4all/xfac


Factorize a (discrete) tensor function
======================================

.. doxygenstruct:: xfac::TensorCI2
    :members:

.. doxygenstruct:: xfac::TensorTrain
    :members:

Factorize a continuous function using natural grid
==================================================

.. doxygenfunction:: xfac::grid::QuadratureGK15

.. doxygenclass:: xfac::CTensorCI2
    :members:

.. doxygenstruct:: xfac::CTensorTrain
    :members:

Factorize a continuous function using quantics grid
===================================================


.. doxygenstruct:: xfac::grid::Quantics
    :members:

.. doxygenclass:: xfac::QTensorCI
    :members:

.. doxygenstruct:: xfac::QTensorTrain
    :members:

Automatic building of matrix product operators (autoMPO)
========================================================

.. doxygennamespace:: xfac::autompo
    :members:

Index
=====

:ref:`genindex`

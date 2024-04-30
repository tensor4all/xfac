(intro)=

# Welcome to xfac's documentation!

This is the c++ API documentation.

## Installation

To install xfac you just have to follow the instructions at the README.md file that you can get from the repo:
https://github.com/tensor4all/xfac

## Tutorials

* [Quantics for 1d function](/tutorial-python/quantics1d)

## Factorize a (discrete) tensor function

```{eval-rst}

.. doxygenstruct:: xfac::TensorCI2
    :members:

.. doxygenstruct:: xfac::TensorTrain
    :members:
```

## Factorize a continuous function using natural grid

```{eval-rst}

.. doxygenfunction:: xfac::grid::QuadratureGK15

.. doxygenclass:: xfac::CTensorCI2
    :members:

.. doxygenstruct:: xfac::CTensorTrain
    :members:
```

## Factorize a continuous function using quantics grid

```{eval-rst}

.. doxygenstruct:: xfac::grid::Quantics
    :members:

.. doxygenclass:: xfac::QTensorCI
    :members:

.. doxygenstruct:: xfac::QTensorTrain
    :members:
```

## Automatic building of matrix product operators (autoMPO)

```{eval-rst}

.. doxygennamespace:: xfac::autompo
    :members:
```

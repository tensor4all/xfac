(intro)=

# Welcome to xfac's documentation!

Xfac is a numerical software library to learn low-rank tensor train representations.
It is written in C++, provides Python bindings and is distributed under an open-source license.
This site collects Python tutorial examples and the documentation of the C++ API.
The algorithms and the mathematical background are described in reference:

> Yuriel Núñez Fernández, Marc K. Ritter, Matthieu Jeannin, Jheng-Wei Li, Thomas Kloss, Olivier Parcollet, Jan von Delft, Hiroshi Shinaoka, and Xavier Waintal, 
> *Learning low-rank tensor train representations: new algorithms and libraries*, *in preparation*, (2024).


**Source code**

The public source code repo is: https://github.com/tensor4all/xfac

**Installation**

Install Xfac according to the instructions in the [README.md](https://github.com/tensor4all/xfac/blob/main/README.md) file of the code repo.

**License**

The licence is written in file [LICENSE.rst](https://github.com/tensor4all/xfac/blob/main/LICENSE.rst) of the code repo.


## Tutorials

* [Quantics for 1d function](/tutorial-python/quantics1d)
* [Quantics for 2d function](/tutorial-python/quantics2d)
* [Integration of a multivariate function](/tutorial-python/integral_nd)

## C++ API documentation

### Factorize a (discrete) tensor function

```{eval-rst}

.. doxygenstruct:: xfac::TensorCI2
    :members:

.. doxygenstruct:: xfac::TensorTrain
    :members:
```

### Factorize a continuous function using natural grid

```{eval-rst}

.. doxygenfunction:: xfac::grid::QuadratureGK15

.. doxygenclass:: xfac::CTensorCI2
    :members:

.. doxygenstruct:: xfac::CTensorTrain
    :members:
```

### Factorize a continuous function using quantics grid

```{eval-rst}

.. doxygenstruct:: xfac::grid::Quantics
    :members:

.. doxygenclass:: xfac::QTensorCI
    :members:

.. doxygenstruct:: xfac::QTensorTrain
    :members:
```

### Automatic building of matrix product operators (autoMPO)

```{eval-rst}

.. doxygennamespace:: xfac::autompo
    :members:
```

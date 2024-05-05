(intro)=

# Welcome to xfac's documentation!


`xfac` is a numerical software library to learn low-rank tensor train representations form a given tensor or function.
The learning is made by *tensor cross interpolation*.
Given a multidimensional function $f:\mathcal{R}^n \rightarrow \mathcal{C}$, the library can generate a factorization:

$$
f(x_1,x_2,...,x_n) \approx M_1(x_1)M_2(x_2)...M_n(x_n)
$$

where $M$ are matrices. As the variables are now separated, some tasks like integration or sum becomes very cheap.
This factorization can be relevant even for function of one variable via the so-called "quantics" representation [^1]. 

As examples of applications we can mention:
1. Integration of multidimensional functions:
    - quadratures
    - quantics
2. Computation of partition functions
3. Quantics: superfast fft
4. Quantics: solving partial differential equations
5. Automatic construction of matrix product operators
6. Function optimization
7. Quantum chemistry


`xfac` is written in C++, provides Python bindings and is distributed under an open-source license.
This site collects Python tutorial examples and the documentation of the C++ API.
The algorithms and the mathematical background are described in reference {cite}`XfacPaper`.

**Source code**

The public source code repo is: https://github.com/tensor4all/xfac

**Installation**

Install `xfac` according to the instructions in the [README.md](https://github.com/tensor4all/xfac/blob/main/README.md) file of the code repo.

**License**

The licence is written in file [LICENSE.rst](https://github.com/tensor4all/xfac/blob/main/LICENSE.rst) of the code repo.


```{bibliography}
```

[^1]: In quantics, each variable is replaced by its binary digits, making explicit the possible scale separation of the function.
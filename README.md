# Cross factorization of tensors (xfac)

Variable separation to approximate a given tensor or function using tensor train cross interpolation.

## Documentation

- API documentation and tutorials are kindly hosted at https://xfac.readthedocs.io/en/latest
- You can also read the code because is not big and includes comments :-)

### Examples

See folder `example` (for **c++**) and `notebook` (for **python**).

## Dependencies

The dependencies are **automatically satisfied** using `git submodule`. They are

- [armadillo](http://arma.sourceforge.net/) for linear algebra on matrix and 3-leg tensor. Armadillo depends on **blas**, **lapack**.
- [Catch2](https://github.com/catchorg/Catch2) for testing.
- [pybind11](https://github.com/pybind/pybind11) for python interface.
- [carma](https://github.com/RUrlus/carma) to convert armadillo objects to python numpy array.

## Installation

```
git clone https://gitlab.kwant-project.org/ttd/xfac.git
cd xfac
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
```

If you want:

* To compile the **tests**, add `-D XFAC_BUILD_TEST=ON` to  the cmake line above.

* To generate the **python** library, add `-D XFAC_BUILD_PYTHON=ON`

* To **install** the library in specific **local directory**, add `-D CMAKE_INSTALL_PREFIX=<my_local_install_directory>`

To compile and install (respectively):
```
cmake --build build -j 4
cmake --build build --target install
```
the last line is optional and may need admin rights (`sudo`) if no local directory was provided.

#### Compilation on macOS

Compilation under macOS has been tested using the `gcc` compiler. `gcc` itself can be installed with `Homebrew`.
To build `xfac` using this compiler, the two environment variables `CC` and `CXX` must point to `gcc`'s C and respectively C++ compiler binary.
Having for instance `gcc` version 13 installed, one typically needs to add
```
export CC=/usr/local/bin/gcc-13
export CXX=/usr/local/bin/g++-13
```
to the `~/.zshrc` script. Above compiler path is the standard path where
`gcc` is usually found under macOS. If `gcc` is installed in a different location or has a different version, the path must be modified accordingly.

Moreover, the `git submodule` mechanism should ensure that the other dependecies are satisfied automatically. Otherwise, if one likes to use own
libraries, one can point to them as well in the `~/.zshrc` script. For Armadillo for instance, this would look like

```
export PATH=YOUR_PATH_TO_ARMADILLOD/include:$PATH
```
where `YOUR_PATH_TO_ARMADILLOD` must be set to the correct path.


## Usage

A simple example like `integral.cpp` at `example/integral/` can be compiled directly (after building `xfac`):
```
g++ -O3 integral.cpp -std=c++17 -L../../build -lxfac -L../../build/extern/arma -larmadillo -I../../include
```
by manually specifying the path to `xfac` and `armadillo` libraries.

### With cmake

If you use `cmake` for your project, then `xfac` can be used after building without exporting any path (hopefully), by adding this to your `CMakeLists.txt`

```
find_package(xfac REQUIRED)
target_link_libraries(myTarget xfac::xfac)
```

If you experience any problem, you can try the above combined with a later call to:

```
cmake <my_stuff> -D CMAKE_INSTALL_PREFIX=<my_local_directory>
```

where `<my_local_directory>` can be `<my_path_to_xfac>/build` (then you are linking the built library) or `<my_local_install_directory>` in case you installed `xfac` on a local directory.


#### including source code

Alternatively, you can manually copy the `xfac` source code inside your project and add `xfac` as subproject in your `CMakeLists.txt`:
```
add_directory(xfac)
target_link_library(myTarget xfac::xfac)
```
Notice that the manual copy can be automatized by adding `xfac` as a `git submodule` or using `cmake FetchContent`.


## Building the documentation
At the folder docs:
- The [Doxygen](https://doxygen.nl) documentation can be generate by typing `doxygen`. The output goes to `doxygen_out/` including the html version (open `index.html`) and latex version (after `make` you will find refman.pdf).

- The [Sphinx](https://www.sphinx-doc.org/) documentation can also be generated (requires `doxygen`, `sphinx_rtd_theme` and [breathe](https://www.breathe-doc.org/)) by doing:
```
sphinx-build -b html . doc_out
```

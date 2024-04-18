rm -rf build/
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D XFAC_BUILD_TEST=ON -D XFAC_BUILD_PYTHON=ON
cmake --build build -j 4

cmake_minimum_required(VERSION 3.15)

project(xfac_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(autoMPO/)
add_subdirectory(integral/)
add_subdirectory(ising/)
add_subdirectory(quantics1d/)
add_subdirectory(quantics2d/)
add_subdirectory(quanticsFourier/)

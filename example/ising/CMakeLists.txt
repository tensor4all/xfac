cmake_minimum_required(VERSION 3.15)

project(ising LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(ising ising.cpp)

find_package(xfac REQUIRED)
target_link_libraries(ising xfac::xfac)


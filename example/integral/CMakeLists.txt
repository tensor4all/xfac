cmake_minimum_required(VERSION 3.15)

project(integral LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(integral integral.cpp)

find_package(xfac REQUIRED)
target_link_libraries(integral xfac::xfac)


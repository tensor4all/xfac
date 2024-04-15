cmake_minimum_required(VERSION 3.15)

project(quantics2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(${PROJECT_NAME} quantics2d.cpp)

find_package(xfac REQUIRED)
target_link_libraries(${PROJECT_NAME} xfac::xfac)


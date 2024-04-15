cmake_minimum_required(VERSION 3.15)

project(autoMPO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(${PROJECT_NAME} autoMPO.cpp)

find_package(xfac REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE xfac::xfac)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall ${OpenMP_CXX_FLAGS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()


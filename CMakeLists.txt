cmake_minimum_required(VERSION 3.15)
project(xfac LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(XFAC_BUILD_TEST "Build the xfac test suite" OFF)
option(XFAC_BUILD_PYTHON "Build the xfac python library" OFF)

set(header
    include/xfac/index_set.h
    include/xfac/matrix/mat_decomp.h
    include/xfac/matrix/adaptive_lu.h
    include/xfac/matrix/cross_data.h
    include/xfac/matrix/matrix_interface.h
    include/xfac/matrix/pivot_finder.h
    include/xfac/tensor/tensor_train.h
    include/xfac/tensor/tensor_function.h
    include/xfac/tensor/tensor_ci.h
    include/xfac/tensor/tensor_ci_2.h
    include/xfac/tensor/auto_mpo.h
    include/xfac/tensor/tensor_ci_converter.h
    include/xfac/grid.h
    include/xfac/tree/tree.h
)

add_library(${PROJECT_NAME} STATIC
    README.md
    empty.cpp
    ${header}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

#---------- external dependencies ---------

include(extern/extern.cmake)

target_link_libraries(${PROJECT_NAME} PUBLIC armadillo)

if(OpenMP_CXX_FOUND)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall ${OpenMP_CXX_FLAGS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()

#----------- tests -----------------

if (XFAC_BUILD_TEST)
    add_subdirectory(test)
endif()

#---------- pybinding ---------------

if (XFAC_BUILD_PYTHON)
    add_subdirectory(python/)
endif()


#----------------------- install ----------------- 

# https://cmake.org/cmake/help/v3.25/guide/importing-exporting/index.html#exporting-targets

install(TARGETS xfac armadillo
    EXPORT xfacTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/xfac DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT xfacTargets
    FILE xfacTargets.cmake
    NAMESPACE xfac::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xfac
)

add_library(xfac::xfac STATIC IMPORTED)

set_target_properties(xfac::xfac PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/xfacConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xfac
)

install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/xfacConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xfac
)

#----------------------- export -------------------

export(EXPORT xfacTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/xfacTargets.cmake"
       NAMESPACE xfac::
)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE xfac)


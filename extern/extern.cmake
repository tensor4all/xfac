find_package(OpenMP)


find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/arma/CMakeLists.txt")
    message(FATAL_ERROR "Please download git submodules and try again.")
endif()

add_subdirectory(extern/arma)

if (XFAC_BUILD_TEST)
    add_subdirectory(extern/Catch2)
endif()

if (XFAC_BUILD_PYTHON)
    add_subdirectory(extern/pybind11)
    add_subdirectory(extern/carma)
endif()

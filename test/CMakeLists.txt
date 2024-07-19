add_executable(${PROJECT_NAME}_test 
    test_main.cpp
    test_matrix_ci.cpp
    test_matrix_ci_2.cpp
    test_tensor_ci.cpp
    test_tensor_ci_2.cpp
    test_timing.cpp
    test_ci_converter.cpp
    test_tree.cpp
)

target_include_directories(${PROJECT_NAME}_test INTERFACE Catch2::Catch2)
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_test PRIVATE Catch2::Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib/)
include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME}_test)


file(COPY ${CMAKE_SOURCE_DIR}/test/qFP.txt
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_SOURCE_DIR}/test/qFP_1.txt
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

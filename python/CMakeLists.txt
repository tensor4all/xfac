pybind11_add_module(${PROJECT_NAME}py
    python_bindings.cpp
)
target_link_libraries(${PROJECT_NAME}py PUBLIC ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}py PUBLIC carma::carma)

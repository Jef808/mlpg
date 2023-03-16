set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(mlpg_data)
add_subdirectory(Data)

project(mlpg_ann)
add_subdirectory(ann)

project(mlpg_tests)
add_subdirectory(tests)

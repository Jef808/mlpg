set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/${CMAKE_PROJECT_NAME}_tests)

set(mlpg_Ann_tests_dependencies
  mlpg::PCH
  #mlpg::Ann
  mlpg::Data
  #Eigen3::Eigen
  spdlog::spdlog
  Catch2::Catch2WithMain
)

set(CMAKE_CXX_STANDARD 20)

add_executable(test_serialize test_serialize.cpp)
target_link_libraries(test_serialize PRIVATE ${mlpg_Ann_tests_dependencies})
catch_discover_tests(test_serialize)

add_executable(test_function_data testDataGen.cpp)
target_link_libraries(test_function_data PRIVATE ${mlpg_Ann_tests_dependencies})

add_executable(test_ann_function testAnnFunction.cpp)
target_link_libraries(test_ann_function PRIVATE ${mlpg_Ann_tests_dependencies})

add_executable(test_perceptron test_perceptron.cpp)
target_link_libraries(test_perceptron PRIVATE ${mlpg_Ann_tests_dependencies})

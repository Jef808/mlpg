project(mcts-simple VERSION 1.0)

add_executable(test_ann
  ${CMAKE_CURRENT_SOURCE_DIR}/testLinearAnn.cpp)
target_link_libraries(test_ann PUBLIC
  ann
  Eigen3::Eigen
  )
target_include_directories(test_ann PUBLIC
  ${CMAKE_SOURCE_DIR}/src/ann
  )


add_library(function_ex INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionExampleData.h)
target_link_libraries(function_ex INTERFACE
  Eigen3::Eigen)
target_include_directories(function_ex INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR})


add_executable(test_function_data
  ${CMAKE_CURRENT_SOURCE_DIR}/testDataGen.cpp)
target_link_libraries(test_function_data PUBLIC
  function_ex)

add_executable(test_ann_function
  ${CMAKE_CURRENT_SOURCE_DIR}/testAnnFunction.cpp)
target_link_libraries(test_ann_function PUBLIC
  function_ex
  ann2
  Eigen3::Eigen)

add_executable(test_ann_mnist
  ${CMAKE_CURRENT_SOURCE_DIR}/testMnist.cpp)
target_link_libraries(test_ann_mnist PUBLIC
  dataIO
  ann2
  Eigen3::Eigen)

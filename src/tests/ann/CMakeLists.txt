#find_package(fmt)

add_executable(test_ann
  ${CMAKE_CURRENT_SOURCE_DIR}/testLinearAnn.cpp)
target_link_libraries(test_ann PUBLIC ann Eigen3::Eigen)# fmt::fmt)
target_include_directories(test_ann PUBLIC ${CMAKE_SOURCE_DIR}/src/ann)

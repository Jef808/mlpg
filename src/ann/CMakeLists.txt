add_library(ann ${CMAKE_CURRENT_SOURCE_DIR}/LinearAnn.cpp)
target_link_libraries(ann PUBLIC Eigen3::Eigen)
target_include_directories(ann PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

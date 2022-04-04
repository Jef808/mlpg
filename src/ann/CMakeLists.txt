# add_library(ann ${CMAKE_CURRENT_SOURCE_DIR}/LinearAnn.cpp)
# target_link_libraries(ann PUBLIC Eigen3::Eigen)
# target_include_directories(ann PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# set_property(TARGET ann PROPERTY CXX_STANDARD 17)

add_library(ann INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/TemplateAnn.h)
target_link_libraries(ann INTERFACE
  dataIO
  Eigen3::Eigen)
target_include_directories(ann INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/data)
set_property(TARGET ann
  PROPERTY CXX_STANDARD 17)

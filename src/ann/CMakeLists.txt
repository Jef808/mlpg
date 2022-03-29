# add_library(ann ${CMAKE_CURRENT_SOURCE_DIR}/LinearAnn.cpp)
# target_link_libraries(ann PUBLIC Eigen3::Eigen)
# target_include_directories(ann PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# set_property(TARGET ann PROPERTY CXX_STANDARD 17)

add_library(ann2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/TemplateAnn.h)
target_link_libraries(ann2 INTERFACE Eigen3::Eigen)
target_include_directories(ann2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET ann2 PROPERTY CXX_STANDARD 17)

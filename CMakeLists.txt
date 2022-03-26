cmake_minimum_required(VERSION 3.20)
project(mcts-simple VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

list(PREPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  /usr/local/share/eigen3/cmake/)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup(TARGETS)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)



set(BUILD_VIEWER ON)

if (BUILD_VIEWER)
  include(imgui-glfw-glad-glm)
endif()



add_subdirectory(src)

cmake_minimum_required(VERSION 3.20)
project(mcts-simple VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

list(PREPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  /usr/local/share/eigen3/cmake/)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

#include(eigen)
add_subdirectory(src)

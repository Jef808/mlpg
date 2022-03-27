cmake_minimum_required(VERSION 3.20)
project(mcts-simple VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

list(PREPEND CMAKE_MODULE_PATH
  /usr/local/share/eigen3/cmake/
  /home/jfa/.local/lib/cmake/spdlog/
  ${CMAKE_SOURCE_DIR}/cmake/
  )


find_package(Eigen3 3.3 REQUIRED NO_MODULE)

message(STATUS "cmake module path is \n${CMAKE_MODULE_PATH}")


set(spdlog_DIR ${CMAKE_SOURCE_DIR}/3rdparty/spdlog/)
include_directories(${CMAKE_SOURCE_DIR}/lib/include)
find_package(spdlog REQUIRED)



set(BUILD_VIEWER ON)

if (BUILD_VIEWER)
  include(imgui-glfw-glad-glm)
endif()



add_subdirectory(src)

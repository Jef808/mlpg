cmake_minimum_required(VERSION 3.20)
project(mcts-simple VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Defines variables for logging with colors from cmake
add_subdirectory(scripts)

list(PREPEND 
  CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/
  /usr/local/share/eigen3/cmake/
  /home/jfa/.local/lib/cmake/spdlog/
  ${CMAKE_SOURCE_DIR}/3rdparty/sanitizers-cmake/cmake/
  )

set(BUILD_SHARED_LIBS OFF)

include(imgui-glfw-glad-glm)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

message(STATUS "cmake module path is \n${CMAKE_MODULE_PATH}")


set(spdlog_DIR ${CMAKE_SOURCE_DIR}/3rdparty/spdlog/)
include_directories(${CMAKE_SOURCE_DIR}/lib/include)
find_package(spdlog REQUIRED)


set(BUILD_VIEWER ON)

if (${BUILD_VIEWER})

endif()



add_subdirectory(src)

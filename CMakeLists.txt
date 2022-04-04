cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(_project_options
  URL https://github.com/aminya/project_options/archive/refs/tags/v0.20.0.zip)
FetchContent_MakeAvailable(_project_options)

include(${_project_options_SOURCE_DIR}/Index.cmake)

project(mcts-simple VERSION 1.0 LANGUAGES CXX C)

# Initialize project_options variable related to this project
# This overwrites `project_options` and sets `project_warnings`
# uncomment to enable the options. Some of them accept one or more inputs:
project_options(
      ENABLE_CACHE
      # ENABLE_CPPCHECK
      # ENABLE_CLANG_TIDY
      # ENABLE_CONAN
      ENABLE_INTERPROCEDURAL_OPTIMIZATION
      ENABLE_NATIVE_OPTIMIZATION
      ENABLE_DOXYGEN
      # ENABLE_COVERAGE
      ENABLE_SANITIZER_ADDRESS
      # ENABLE_SANITIZER_LEAK
      ENABLE_SANITIZER_UNDEFINED_BEHAVIOR
      # ENABLE_SANITIZER_THREAD
      # ENABLE_SANITIZER_MEMORY
      # ENABLE_PCH
      # PCH_HEADERS
      # WARNINGS_AS_ERRORS
      # ENABLE_INCLUDE_WHAT_YOU_USE
      # ENABLE_USER_LINKER
      # ENABLE_BUILD_WITH_TIME_TRACE
      # ENABLE_UNITY
      # CONAN_OPTIONS
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Defines variables for logging with colors from cmake
add_subdirectory(scripts)

list(
  PREPEND
  CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/
  /usr/local/share/eigen3/cmake/
  /home/jfa/.local/lib/cmake/spdlog/
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

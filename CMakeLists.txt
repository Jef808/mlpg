cmake_minimum_required(VERSION 3.22)
project(mlpg VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(FetchContent)

FetchContent_Declare(
  _project_options
  URL https://github.com/aminya/project_options/archive/refs/tags/v0.20.0.zip)
FetchContent_MakeAvailable(_project_options)

include(${_project_options_SOURCE_DIR}/Index.cmake)


# Initialize project_options variable related to this project
# This overwrites `project_options` and sets `project_warnings`
# uncomment to enable the options. Some of them accept one or more inputs:
project_options(
      # ENABLE_CACHE
      ENABLE_CPPCHECK
      # ENABLE_CLANG_TIDY
      # ENABLE_CONAN
      # ENABLE_INTERPROCEDURAL_OPTIMIZATION
      # ENABLE_NATIVE_OPTIMIZATION
      # ENABLE_DOXYGEN
      # ENABLE_COVERAGE
      ENABLE_SANITIZER_ADDRESS
      ENABLE_SANITIZER_LEAK
      ENABLE_SANITIZER_UNDEFINED_BEHAVIOR
      # ENABLE_SANITIZER_THREAD
      ENABLE_SANITIZER_MEMORY
      ENABLE_PCH
      # PCH_HEADERS
      # WARNINGS_AS_ERRORS
      # ENABLE_INCLUDE_WHAT_YOU_USE
      # ENABLE_USER_LINKER
      # ENABLE_BUILD_WITH_TIME_TRACE
      # ENABLE_UNITY
      # CONAN_OPTIONS
)

# set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)


include(eigen)
include(imgui-glfw-glad-glm)
include(spdlog)
include(implot)

include(Catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

enable_testing()

# set(spdlog_DIR ${EXTERNAL_DIR}/spdlog)
# find_package(spdlog REQUIRED)

set(mlpg_DATA_DIR ${CMAKE_SOURCE_DIR}/src/Data/include/Data/)

add_library(mlpg_pch INTERFACE)
target_include_directories(mlpg_pch INTERFACE ${CMAKE_SOURCE_DIR}/src/ann/include)
target_link_libraries(mlpg_pch INTERFACE project_options project_warnings Eigen3::Eigen)
target_precompile_headers(mlpg_pch INTERFACE src/ann/include/ann/TemplateAnn.h)
target_compile_definitions(mlpg_pch INTERFACE "-Dfopenmp")

add_library(mlpg::PCH ALIAS mlpg_pch)

add_subdirectory(src)

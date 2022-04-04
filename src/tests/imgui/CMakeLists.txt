project(mcts-simple VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

add_executable(testimgui testimgui.cpp)

target_link_libraries(
  testimgui
  imgui
  glm
  )

find_package(Threads)

add_executable(testannimgui
    testannimgui.cpp
    ${CMAKE_SOURCE_DIR}/3rdparty/implot/implot.cpp
    ${CMAKE_SOURCE_DIR}/3rdparty/implot/implot_items.cpp
)
# add_sanitizers(testannimgui)

target_link_libraries(
  testannimgui
  imgui
  glm
  Eigen3::Eigen
  Threads::Threads
)
target_include_directories(
    testannimgui PRIVATE
    ${CMAKE_SOURCE_DIR}/3rdparty/implot/
)

set(MY_DEBUG_COMPILE_OPTIONS
  "SHELL:-g"
  "SHELL:-fsanitize=address"
  "SHELL:-fno-omit-frame-pointer"
  "SHELL:-O1")
set(MY_DEBUG_LINK_OPTIONS
  "SHELL:-fsanitize=address"
  "SHELL:-fno-omit-frame-pointer")
set(MY_RELEASE_COMPILE_OPTIONS
  "SHELL:-ffast-math"
  "SHELL:-march=native"
  "SHELL:-DNDEBUG")
target_compile_options(testannimgui PUBLIC
  "$<$<CONFIG:DEBUG>:${MY_DEBUG_COMPILE_OPTIONS}>")
target_compile_options(testannimgui PRIVATE
  "$<$<CONFIG:RELEASE>:${MY_RELEASE_COMPILE_OPTIONS}>")
target_link_options(testannimgui PUBLIC
  "$<$<CONFIG:DEBUG>:${MY_DEBUG_LINK_OPTIONS}>")

include(CMakePrintHelpers)

cmake_print_properties(
  TARGETS testannimgui
  PROPERTIES COMPILE_OPTIONS LINK_OPTIONS
)

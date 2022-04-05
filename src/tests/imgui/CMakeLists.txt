project(mcts-simple VERSION 1.0)

# add_executable(testimgui testimgui.cpp)
# target_link_libraries(
#   testimgui
#   imgui
#   glm
# )

set( IMPLOT_SOURCES
  ${CMAKE_SOURCE_DIR}/external/implot/implot.cpp
  ${CMAKE_SOURCE_DIR}/external/implot/implot_items.cpp
  )
set( IMPLOT_HEADERS
  ${EXTERNAL_DIR}/implot/implot.h
  ${EXTERNAL_DIR}/implot/implot_internal.h
  )
set( IMPLOT_INCLUDE_DIRS ${EXTERNAL_DIR}/implot )
set( IMPLOT_DEPENDENCIES imgui glm )


add_library( implot
  ${IMPLOT_SOURCES})
target_link_libraries( implot PUBLIC
  ${IMPLOT_DEPENDENCIES} )
target_include_directories( implot PUBLIC
  ${IMPLOT_INCLUDE_DIRS} )


find_package(Threads REQUIRED)
#find_package(Eigen3 REQUIRED)

add_executable(
  basicannimgui
  basicannimgui.cpp
  # ${CMAKE_SOURCE_DIR}/external/implot/implot.cpp
  # ${CMAKE_SOURCE_DIR}/external/implot/implot_items.cpp
)

# link project_options/warnings
target_link_libraries(basicannimgui PRIVATE project_options project_warnings)
# Set sources
target_link_system_libraries(
  basicannimgui
  # PUBLIC
  # ${LOCAL_SOURCES_CONFIGURED}
  PRIVATE
  implot
  Threads::Threads
)

target_link_libraries(basicannimgui
  PRIVATE
  ann)


# Add logs
include(CMakePrintHelpers)

cmake_print_properties(
  TARGETS
  basicannimgui
  PROPERTIES
  COMPILE_OPTIONS
)

# set(
#   MY_DEBUG_COMPILE_OPTIONS
#   "SHELL:-g"
#   "SHELL:-fsanitize=address"
#   "SHELL:-fno-omit-frame-pointer"
#   "SHELL:-O1"
# )
# set(
#   MY_DEBUG_LINK_OPTIONS
#   "SHELL:-fsanitize=address"
#   "SHELL:-fno-omit-frame-pointer"
# )
# set(
#   MY_RELEASE_COMPILE_OPTIONS
#   "SHELL:-ffast-math"
#   "SHELL:-march=native"
#   "SHELL:-DNDEBUG"
# )
# target_compile_options(
#   basicannimgui PUBLIC
#   "$<$<CONFIG:DEBUG>:${MY_DEBUG_COMPILE_OPTIONS}>"
# )
# target_compile_options(
#   basicannimgui PRIVATE
#   "$<$<CONFIG:RELEASE>:${MY_RELEASE_COMPILE_OPTIONS}>"
# )
#target_link_options(
#  basicannimgui PUBLIC
#  "$<$<CONFIG:DEBUG>:${MY_DEBUG_LINK_OPTIONS}>"
#)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/${CMAKE_PROJECT_NAME}_tests)

set( IMPLOT_SOURCES
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  )
set( IMPLOT_HEADERS
  ${implot_SOURCE_DIR}/implot.h
  ${implot_SOURCE_DIR}/implot_internal.h
  )
# set( IMPLOT_SOURCES
#   ${CMAKE_SOURCE_DIR}/external/implot/implot.cpp
#   ${CMAKE_SOURCE_DIR}/external/implot/implot_items.cpp
#   )
# set( IMPLOT_HEADERS
#   ${EXTERNAL_DIR}/implot/implot.h
#   ${EXTERNAL_DIR}/implot/implot_internal.h
#   )

set( IMPLOT_INCLUDE_DIRS ${implot_SOURCE_DIR} )
set( IMPLOT_DEPENDENCIES imgui glm )


add_library( implot
  ${IMPLOT_SOURCES})
target_link_libraries( implot PUBLIC
  ${IMPLOT_DEPENDENCIES} )
target_include_directories( implot PUBLIC
  ${IMPLOT_INCLUDE_DIRS} )


find_package(Threads REQUIRED)

add_executable(
  basicannimgui
  basicannimgui.cpp
)

# link project_options/warnings
target_link_libraries(
  basicannimgui
  PRIVATE
  project_options project_warnings
  mlpg::Ann
  mlpg::Data
  implot
  Threads::Threads
)

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

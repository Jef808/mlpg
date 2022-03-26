project(viewer VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native)


###############################################################################
# 3RD PARTY LIBS
###############################################################################

include(FetchContent)

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt)
FetchContent_MakeAvailable(fmt)

set(BUILD_SHARED_LIBS OFF)




###############################################################################
# IMGUI
###############################################################################

set( MY_IMGUI_ROOT $ENV{HOME}/repos/imguiroot)

set(MY_IMGUI_HEADERS
    ${MY_IMGUI_ROOT}/imgui/imconfig.h
    ${MY_IMGUI_ROOT}/imgui/imgui_internal.h
    ${MY_IMGUI_ROOT}/imgui/imgui.h
    ${MY_IMGUI_ROOT}/imgui/imstb_rectpack.h
    ${MY_IMGUI_ROOT}/imgui/imstb_textedit.h
    ${MY_IMGUI_ROOT}/imgui/imstb_truetype.h
    ${MY_IMGUI_ROOT}/imgui/backends/imgui_impl_glfw.h
    ${MY_IMGUI_ROOT}/imgui/backends/imgui_impl_opengl3.h
    ${MY_IMGUI_ROOT}/imgui/backends/imgui_impl_opengl3_loader.h
    ${MY_IMGUI_ROOT}/imgui/misc/cpp/imgui_stdlib.h
)

set(MY_IMGUI_SRC
    ${MY_IMGUI_ROOT}/imgui/imgui_demo.cpp
    ${MY_IMGUI_ROOT}/imgui/imgui_draw.cpp
    ${MY_IMGUI_ROOT}/imgui/imgui_tables.cpp
    ${MY_IMGUI_ROOT}/imgui/imgui_widgets.cpp
    ${MY_IMGUI_ROOT}/imgui/imgui.cpp
    ${MY_IMGUI_ROOT}/imgui/backends/imgui_impl_glfw.cpp
    ${MY_IMGUI_ROOT}/imgui/backends/imgui_impl_opengl3.cpp
    ${MY_IMGUI_ROOT}/imgui/misc/cpp/imgui_stdlib.cpp
)

set(MY_IMGUI_INCLUDE_DIRS ${MY_IMGUI_ROOT}/imgui)

message(STATUS "Imgui Headers and sources:" ${MY_IMGUI_HEADERS} ${MY_IMGUI_SRC})


find_package(OpenGL REQUIRED)

#add_library(imgui ${IMGUI_HEADERS} ${IMGUI_SRC})






###############################################################################
# IMPLOT
###############################################################################


set(MY_IMPLOT_HEADERS
  ${MY_IMGUI_ROOT}/implot/implot.h
  ${MY_IMGUI_ROOT}/implot/implot_internal.h
)

set(MY_IMPLOT_SRC
  ${MY_IMGUI_ROOT}/implot/implot.cpp
  ${MY_IMGUI_ROOT}/implot/implot_items.cpp
  ${MY_IMGUI_ROOT}/implot/implot_demo.cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/implot/backends")
  message(STATUS "Found the implot backends in current directory")
  list(APPEND IMPLOT_HEADERS
    ${MY_IMGUI_ROOT}/implot/backends/implot_backend.h
    ${MY_IMGUI_ROOT}/implot/backends/implot_impl_opengl3.h
  )
  list(APPEND IMPLOT_SRC
    ${MY_IMGUI_ROOT}/implot/backends/implot_backend.h
    ${MY_IMGUI_ROOT}/implot/backends/implot_impl_opengl3.cpp
  )
endif()


message(STATUS "Implot Headers and sources:" ${MY_IMPLOT_HEADERS} ${MY_IMPLOT_SRC})


###############################################################################
# APP FRAMEWORK
###############################################################################


# add_library(app
#   ${MY_IMGUI_ROOT}/implot_demos/common/App.h
#   ${MY_IMGUI_ROOT}/implot_demos/common/App.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Shader.h
#   ${MY_IMGUI_ROOT}/implot_demos/common/Native.h
#   ${MY_IMGUI_ROOT}/implot_demos/common/Native.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Image.h
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/Fonts.h
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/FontAwesome5.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/RobotoRegular.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/RobotoBold.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/RobotoItalic.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/RobotoMonoRegular.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/RobotoMonoBold.cpp
#   ${MY_IMGUI_ROOT}/implot_demos/common/Fonts/RobotoMonoItalic.cpp
# )
# target_include_directories(app PUBLIC ${MY_IMGUI_ROOT}/implot_demos/common)
# #target_link_libraries(app implot nfd)
# target_compile_features(app PRIVATE cxx_std_17)


add_library(implot ${MY_IMPLOT_HEADERS} ${MY_IMPLOT_SRC})
set_property(TARGET implot PROPERTY CXX_STANDARD 11)
target_link_libraries(implot PUBLIC imgui OpenGL::GL)
target_compile_definitions(implot PUBLIC IMPLOT_DEBUG IMPLOT_DLL_EXPORT IMPLOT_BACKEND _ENABLE_OPENGL3 IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_compile_options(implot PRIVATE -Wall -Wextra -pedantic -Werror -mavx2 -Ofast)
target_include_directories(implot PUBLIC
  ${MY_IMGUI_ROOT}/imgui
  ${MY_IMGUI_ROOT}/imgui/backends
  ${MY_IMGUI_ROOT}/implot)



add_executable(viewer
  ${CMAKE_CURRENT_SOURCE_DIR}/example.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui/use_imgui.cpp
  ${MY_IMGUI_HEADERS}
  ${MY_IMGUI_SRC}
  ${MY_IMPLOT_HEADERS}
  ${MY_IMPLOT_SRC}
  )
target_link_libraries(viewer PUBLIC implot)# implot glad glfw OpenGL::GL)
target_include_directories(viewer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui
  ${MY_IMGUI_ROOT}/imgui
  ${MY_IMGUI_ROOT}/imgui/backends
  ###
  ${MY_IMGUI_ROOT}/implot/
)

target_compile_features(viewer PRIVATE cxx_std_17)


############################################################
# PREVIOUS SETUP USING CONAN
############################################################
# add_executable(viewer
#   "${CMAKE_CURRENT_SOURCE_DIR}/example.cpp"
#   "${CMAKE_CURRENT_SOURCE_DIR}/imgui/use_imgui.cpp"
#   #"${CMAKE_CURRENT_SOURCE_DIR}/imgui/use_implot.cpp"
#   "${CONAN_SRC_DIRS_IMGUI}/imgui_impl_glfw.cpp"
#   "${CONAN_SRC_DIRS_IMGUI}/imgui_impl_opengl3.cpp")
#   #"${CONAN_BUILD_DIRS_IMPLOT}/implot.cpp"
#   #"${CONAN_BUILD_DIRS_IMPLOT}/implot_items.cpp")
#   #"${CONAN_SRC_DIRS_IMPLOT}/implot_demo.cpp")
# target_link_libraries(viewer-demo PUBLIC CONAN_PKG::glad CONAN_PKG::glfw CONAN_PKG::imgui)
# target_include_directories(viewer-demo PUBLIC
#   "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
#   "${CONAN_INCLUDE_DIRS_IMGUI}"
#   "${CONAN_SRC_DIRS_IMGUI}")
#   #"${CONAN_INCLUDE_DIRS_IMPLOT}"
#   #"${CONAN_SRC_DIRS_IMPLOT}")

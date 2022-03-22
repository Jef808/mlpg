add_executable(viewer-demo
  "${CMAKE_CURRENT_SOURCE_DIR}/example.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/use_imgui.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/use_implot.cpp"
  "${CONAN_SRC_DIRS_IMGUI}/imgui_impl_glfw.cpp"
  "${CONAN_SRC_DIRS_IMGUI}/imgui_impl_opengl3.cpp")
target_link_libraries(viewer-demo PUBLIC CONAN_PKG::glad CONAN_PKG::glfw CONAN_PKG::imgui CONAN_PKG::implot)
target_include_directories(viewer-demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imgui ${CONAN_INCLUDE_DIRS_IMGUI} ${CONAN_SRC_DIRS_IMGUI})

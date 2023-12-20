include(FetchContent)

FetchContent_Declare(
  imgui-glfw-glad-glm
  GIT_REPOSITORY https://github.com/cmmw/imgui-glfw-glad-glm.git
  GIT_TAG v4.1.0
)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

FetchContent_MakeAvailable(imgui-glfw-glad-glm)

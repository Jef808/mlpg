include(FetchContent)

FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot
  GIT_TAG v0.14
  )

FetchContent_MakeAvailable(implot)

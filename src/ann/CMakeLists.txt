set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED 20)

set(ann_FILES
  Activations.h
  Config.h
  CostFunctions.h
# Network.h
  Shuffle_data.h
  Serialize.h
  TemplateAnn.h
)

list(
  TRANSFORM
  ann_FILES
  PREPEND
  include/ann/
)

add_library(
  mlpg_Ann
  INTERFACE
  ${ann_FILES}
  )

target_link_libraries(mlpg_Ann
  INTERFACE
  project_options project_warnings
  )
target_link_libraries(
  mlpg_Ann
  INTERFACE
  Eigen3::Eigen
  )
target_include_directories(mlpg_Ann
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# target_compile_options(mlpg_Ann INTERFACE "-fconcepts")

target_compile_definitions(mlpg_Ann
  INTERFACE
  "-DMLPG_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
)

add_library(mlpg::Ann ALIAS mlpg_Ann)

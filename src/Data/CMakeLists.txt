set(mlpg_DATA_FILES
  load_csv.h
  filesystem.h
)

list(TRANSFORM
  mlpg_DATA_FILES
  PREPEND
  include/Data/
)

add_library(mlpg_DATA
  INTERFACE
  ${mlpg_DATA_FILES}
)
target_include_directories(mlpg_DATA
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/)

message( INFO "When defining mlpg::Data interface lib, mlpg_DATA_DIR = ${mlpg_DATA_DIR}" )

target_compile_definitions(mlpg_DATA
  INTERFACE
  "-DMLPG_DATA_DIR=\"${mlpg_DATA_DIR}\""
)

add_library(mlpg::Data ALIAS mlpg_DATA)

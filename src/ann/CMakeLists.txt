project(mcts-simple VERSION 1.0)

add_library(
  ann
  INTERFACE
  )
target_link_libraries(ann
  INTERFACE
  project_options project_warnings
  )
target_link_system_libraries(
  ann
  INTERFACE
  eigenlib
  )
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/"include")
target_include_directories(ann
  SYSTEM
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
  )

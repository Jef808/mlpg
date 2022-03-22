# add_executable(ttt ttt.cpp)
# target_link_libraries(ttt PUBLIC tictactoe)

add_executable(ttt new_ttt.cpp)

add_subdirectory(math)

add_subdirectory(replay)

add_subdirectory(ann)

#target_include_directories(ttt PUBLIC ${UTILS_DIR})

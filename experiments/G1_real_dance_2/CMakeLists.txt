add_executable(g1_real_dance_2 src/g1_real_dance_2.cpp)
target_include_directories(g1_real_dance_2
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(g1_real_dance_2 ${PROJECT_NAME})

add_executable(go2_sim_cz_m0p1 src/go2_sim_cz_-0.1m.cpp)
target_include_directories(go2_sim_cz_m0p1
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_sim_cz_m0p1 ${PROJECT_NAME})

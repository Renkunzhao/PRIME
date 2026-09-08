add_executable(go2_sim_cz_m0p1_mhe src/go2_sim_cz_-0.1m_mhe.cpp)
target_include_directories(go2_sim_cz_m0p1_mhe
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_sim_cz_m0p1_mhe ${PROJECT_NAME})

add_executable(go2_sim_m_p3kg_mhe src/Go2_sim_m_+3kg_mhe.cpp)
target_include_directories(go2_sim_m_p3kg_mhe
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_sim_m_p3kg_mhe ${PROJECT_NAME})

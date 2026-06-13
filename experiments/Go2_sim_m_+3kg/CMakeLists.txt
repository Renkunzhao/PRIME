add_executable(go2_sim_m_p3kg src/Go2_sim_m_+3kg.cpp)
target_include_directories(go2_sim_m_p3kg
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_sim_m_p3kg ${PROJECT_NAME})

add_executable(go2_sim_kunzhao_long_mhe src/Go2_sim_kunzhao_long_mhe.cpp)
target_include_directories(go2_sim_kunzhao_long_mhe
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_sim_kunzhao_long_mhe ${PROJECT_NAME})

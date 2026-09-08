add_executable(go2_sim_kunzhao_mhe src/Go2_sim_kunzhao_mhe.cpp)
target_include_directories(go2_sim_kunzhao_mhe
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_sim_kunzhao_mhe ${PROJECT_NAME})

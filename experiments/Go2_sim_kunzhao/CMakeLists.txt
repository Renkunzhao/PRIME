add_executable(go2_sim_kunzhao src/Go2_sim_kunzhao.cpp)
target_include_directories(go2_sim_kunzhao
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(go2_sim_kunzhao ${PROJECT_NAME})

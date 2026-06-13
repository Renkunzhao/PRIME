add_executable(go2_real_belly_plate_4p6kg src/Go2_real_belly_plate_4.6kg.cpp)
target_include_directories(go2_real_belly_plate_4p6kg
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(go2_real_belly_plate_4p6kg ${PROJECT_NAME})

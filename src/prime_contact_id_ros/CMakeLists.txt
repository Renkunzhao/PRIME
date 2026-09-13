cmake_minimum_required(VERSION 3.16)
project(prime_contact_id_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

get_filename_component(PRIME_ROS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(PRIME_SOURCE_DIR "${PRIME_ROS_ROOT}/core" CACHE PATH "Local PRIME source tree")
if(NOT EXISTS "${PRIME_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "PRIME_SOURCE_DIR does not point to a PRIME source tree: ${PRIME_SOURCE_DIR}")
endif()

set(BUILD_PYTHON_INTERFACE OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_CONTACT_ID_EXPERIMENTS OFF CACHE BOOL "" FORCE)
set(BUILD_WITH_IPOPT OFF CACHE BOOL "" FORCE)
set(PRIME_ROS_ENABLE_MULTITHREADS ON CACHE BOOL "Build local PRIME with Crocoddyl node-level multithreading")
set(PRIME_ROS_NTHREADS 8 CACHE STRING "Default thread count for local PRIME build")
if(PRIME_ROS_ENABLE_MULTITHREADS)
  set(BUILD_WITH_MULTITHREADS ON CACHE BOOL "" FORCE)
  set(BUILD_WITH_NTHREADS "${PRIME_ROS_NTHREADS}")
  set(BUILD_WITH_NTHREADS "${PRIME_ROS_NTHREADS}" CACHE STRING "" FORCE)
endif()

add_subdirectory(${PRIME_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/prime EXCLUDE_FROM_ALL)

set(PRIME_ROS_DEPS
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  diagnostic_msgs
)

function(configure_prime_ros_target target_name)
  target_include_directories(${target_name} PRIVATE
    ${PRIME_SOURCE_DIR}/include
    ${PRIME_SOURCE_DIR}/experiments/common
  )
  target_link_libraries(${target_name} crocoddyl)
  ament_target_dependencies(${target_name} ${PRIME_ROS_DEPS})
  if(PRIME_ROS_ENABLE_MULTITHREADS)
    target_compile_definitions(${target_name} PRIVATE
      CROCODDYL_WITH_MULTITHREADING
      CROCODDYL_WITH_NTHREADS=${PRIME_ROS_NTHREADS}
    )
  endif()
endfunction()

add_executable(prime_moving_window_node src/prime_moving_window_node.cpp)
configure_prime_ros_target(prime_moving_window_node)

add_executable(prime_csv_replay_node src/prime_csv_replay_node.cpp)
configure_prime_ros_target(prime_csv_replay_node)

install(TARGETS
  prime_moving_window_node
  prime_csv_replay_node
  DESTINATION lib/${PROJECT_NAME}
)

configure_file(
  config/Go2_sim_kunzhao_long_mhe_ros.xml.in
  ${CMAKE_CURRENT_BINARY_DIR}/Go2_sim_kunzhao_long_mhe_ros.xml
  @ONLY
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/Go2_sim_kunzhao_long_mhe_ros.xml
  config/prime_inertial_input.yaml
  DESTINATION share/${PROJECT_NAME}/config
)
install(PROGRAMS
  scripts/unitree_prime_state_node.py
  scripts/prime_inertial_input_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

"""Start IEKF, PRIME, and both Beam GT policy adapters together."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_iekf",
                default_value="true",
                description="Start IEKF; set false if /iekf/odom is already provided.",
            ),
            DeclareLaunchArgument(
                "prime_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("prime_contact_id_ros"),
                        "config",
                        "Go2_sim_kunzhao_long_mhe_ros.xml",
                    ]
                ),
                description="PRIME XML with valid local model/source/results paths.",
            ),
            DeclareLaunchArgument(
                "inertial_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("prime_contact_id_ros"),
                        "config",
                        "prime_inertial_input.yaml",
                    ]
                ),
                description="Training-model nominal inertia for the GT policy adapter.",
            ),
            DeclareLaunchArgument(
                "iekf_prefix",
                default_value="",
                description="Optional IEKF process prefix, e.g. taskset -c 1.",
            ),
            # Stop the group if any launched node exits, including the included IEKF.
            RegisterEventHandler(
                OnProcessExit(
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(reason="A PRIME/IEKF pipeline node exited")
                        ),
                    ]
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("iekf_go1"),
                            "launch",
                            "go1_simulation_launch.py",
                        ]
                    )
                ),
                launch_arguments={"prefix": LaunchConfiguration("iekf_prefix")}.items(),
                condition=IfCondition(LaunchConfiguration("start_iekf")),
            ),
            Node(
                package="prime_contact_id_ros",
                executable="unitree_prime_state_node.py",
                name="unitree_prime_state",
                output="screen",
            ),
            Node(
                package="prime_contact_id_ros",
                executable="prime_moving_window_node",
                name="prime_moving_window_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("prime_config"),
                        "base_odom_topic": "/prime/odom",
                        "joint_state_topic": "/prime/joint_states",
                    }
                ],
            ),
            Node(
                package="prime_contact_id_ros",
                executable="prime_inertial_input_node.py",
                name="prime_inertial_input",
                output="screen",
                parameters=[LaunchConfiguration("inertial_config")],
            ),
        ]
    )

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("prime_contact_id_ros"),
                        "config",
                        "Go2_sim_kunzhao_long_mhe_ros.xml",
                    ]
                ),
            ),
            DeclareLaunchArgument("base_odom_topic", default_value="/odom"),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            Node(
                package="prime_contact_id_ros",
                executable="prime_moving_window_node",
                name="prime_moving_window_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config_path,
                        "base_odom_topic": LaunchConfiguration("base_odom_topic"),
                        "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                    }
                ],
            ),
        ]
    )

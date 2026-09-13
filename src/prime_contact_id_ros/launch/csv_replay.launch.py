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
            DeclareLaunchArgument("rate_hz", default_value="200.0"),
            DeclareLaunchArgument("start_idx", default_value="10000"),
            DeclareLaunchArgument("max_samples", default_value="0"),
            Node(
                package="prime_contact_id_ros",
                executable="prime_csv_replay_node",
                name="prime_csv_replay_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config_path,
                        "rate_hz": LaunchConfiguration("rate_hz"),
                        "start_idx": LaunchConfiguration("start_idx"),
                        "max_samples": LaunchConfiguration("max_samples"),
                    }
                ],
            ),
        ]
    )

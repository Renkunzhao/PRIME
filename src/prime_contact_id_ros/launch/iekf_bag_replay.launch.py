"""Replay a recorded Unitree LowState stream through the IEKF only."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    workspace = LaunchConfiguration("workspace")
    bag_path = LaunchConfiguration("bag_path")

    iekf = Node(
        package="iekf_go1",
        executable="simulation_sub",
        name="robot_sub",
        output="screen",
        prefix=LaunchConfiguration("iekf_prefix"),
        parameters=[
            {"use_sim_time": False},
            LaunchConfiguration("iekf_config"),
        ],
    )

    state_logger = Node(
        package="prime_contact_id_ros",
        executable="unitree_prime_state_node.py",
        name="unitree_prime_state",
        output="screen",
        parameters=[
            {
                "log_csv": ParameterValue(
                    LaunchConfiguration("log_csv"), value_type=bool
                ),
                "log_directory": LaunchConfiguration("log_directory"),
                "truncate_logs": ParameterValue(
                    LaunchConfiguration("truncate_logs"), value_type=bool
                ),
            }
        ],
    )

    bag_replay = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--rate",
            LaunchConfiguration("playback_rate"),
            "--delay",
            LaunchConfiguration("playback_delay"),
            "--start-offset",
            LaunchConfiguration("start_offset"),
            "--topics",
            "/lowstate",
            "--disable-keyboard-controls",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "workspace",
                default_value=EnvironmentVariable(
                    "WORKSPACE",
                    default_value="/home/jkang/third_party/PRIME_ros",
                ),
                description="Workspace root used by IEKF model configuration paths.",
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value=PathJoinSubstitution(
                    [
                        workspace,
                        "data_bag",
                        "PRIME-default-4kg-oneside",
                    ]
                ),
                description="ROS 2 bag directory containing /lowstate.",
            ),
            DeclareLaunchArgument(
                "iekf_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("iekf_go1"),
                        "config",
                        "parameters_simulation.yaml",
                    ]
                ),
                description="IEKF parameter YAML to tune.",
            ),
            DeclareLaunchArgument(
                "playback_rate",
                default_value="1.0",
                description="Recorded-data playback speed multiplier.",
            ),
            DeclareLaunchArgument(
                "playback_delay",
                default_value="2.0",
                description="Delay before replay so the IEKF subscription is ready.",
            ),
            DeclareLaunchArgument(
                "start_offset",
                default_value="0.0",
                description="Seconds to skip from the beginning of the bag.",
            ),
            DeclareLaunchArgument(
                "iekf_prefix",
                default_value="",
                description="Optional IEKF process prefix, e.g. taskset -c 1.",
            ),
            DeclareLaunchArgument(
                "log_csv",
                default_value="true",
                description="Write PRIME-compatible state and torque CSV files.",
            ),
            DeclareLaunchArgument(
                "log_directory",
                default_value=PathJoinSubstitution(
                    [workspace, "results", "iekf_bag_replay"]
                ),
                description="Directory for p_sense.csv, v_sense.csv, and tau_sense.csv.",
            ),
            DeclareLaunchArgument(
                "truncate_logs",
                default_value="true",
                description="Replace existing CSV logs when the launch starts.",
            ),
            SetEnvironmentVariable(name="WORKSPACE", value=workspace),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=iekf,
                    on_exit=[EmitEvent(event=Shutdown(reason="IEKF exited"))],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=state_logger,
                    on_exit=[EmitEvent(event=Shutdown(reason="State logger exited"))],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_replay,
                    on_exit=[EmitEvent(event=Shutdown(reason="Bag replay finished"))],
                )
            ),
            iekf,
            state_logger,
            bag_replay,
        ]
    )

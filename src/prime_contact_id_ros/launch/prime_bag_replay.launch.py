"""Replay a Unitree bag through IEKF and online PRIME identification."""

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


def shutdown_when_process_exits(action, reason):
    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=[EmitEvent(event=Shutdown(reason=reason))],
        )
    )


def generate_launch_description():
    workspace = LaunchConfiguration("workspace")

    iekf = Node(
        package="iekf_go1",
        executable="simulation_sub",
        name="robot_sub",
        output={"stdout": "own_log", "stderr": "screen"},
        prefix=LaunchConfiguration("iekf_prefix"),
        parameters=[
            {"use_sim_time": False},
            LaunchConfiguration("iekf_config"),
        ],
    )

    state_adapter = Node(
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

    prime = Node(
        package="prime_contact_id_ros",
        executable="prime_moving_window_node",
        name="prime_moving_window_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("prime_config"),
                "base_odom_topic": "/prime/odom",
                "joint_state_topic": "/prime/joint_states",
                "online_start_idx": 0,
                "save_raw_logs": False,
                "solver_verbose": ParameterValue(
                    LaunchConfiguration("solver_verbose"), value_type=bool
                ),
            }
        ],
    )

    inertial_adapter = Node(
        package="prime_contact_id_ros",
        executable="prime_inertial_input_node.py",
        name="prime_inertial_input",
        output="screen",
        parameters=[LaunchConfiguration("inertial_config")],
    )

    bag_replay = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            LaunchConfiguration("bag_path"),
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
                description="PRIME ROS workspace root.",
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value=PathJoinSubstitution(
                    [workspace, "data_bag", "PRIME-default-4kg-oneside"]
                ),
                description="ROS 2 bag directory containing /lowstate.",
            ),
            DeclareLaunchArgument(
                "start_offset",
                default_value="22.0",
                description="Seconds skipped from the beginning of the bag.",
            ),
            DeclareLaunchArgument("playback_rate", default_value="1.0"),
            DeclareLaunchArgument(
                "playback_delay",
                default_value="2.0",
                description="Delay bag publication until all subscribers are ready.",
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
            ),
            DeclareLaunchArgument("iekf_prefix", default_value=""),
            DeclareLaunchArgument(
                "prime_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("prime_contact_id_ros"),
                        "config",
                        "Go2_real_m_+4kg_kunzhao_mhe_ros.xml",
                    ]
                ),
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
            ),
            DeclareLaunchArgument(
                "solver_verbose",
                default_value="false",
                description="Print FDDP iterations for each PRIME window.",
            ),
            DeclareLaunchArgument(
                "log_csv",
                default_value="false",
                description="Optional diagnostic CSV logging; PRIME does not consume it.",
            ),
            DeclareLaunchArgument(
                "log_directory",
                default_value=PathJoinSubstitution(
                    [workspace, "results", "prime_bag_replay", "input_csv"]
                ),
            ),
            DeclareLaunchArgument("truncate_logs", default_value="true"),
            SetEnvironmentVariable(name="WORKSPACE", value=workspace),
            shutdown_when_process_exits(iekf, "IEKF exited"),
            shutdown_when_process_exits(state_adapter, "State adapter exited"),
            shutdown_when_process_exits(prime, "PRIME exited"),
            shutdown_when_process_exits(
                inertial_adapter, "Inertial output adapter exited"
            ),
            shutdown_when_process_exits(bag_replay, "Bag replay finished"),
            iekf,
            state_adapter,
            prime,
            inertial_adapter,
            bag_replay,
        ]
    )

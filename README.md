# PRIME ROS2 Moving Window Workspace

This workspace provides a ROS2 bridge for running PRIME moving window inertial
identification from live ROS topics, while keeping PRIME as a local source-tree
dependency.

Default local paths:

```text
ROS workspace:   /home/jkang/third_party/PRIME_ros
PRIME source:    /home/jkang/third_party/PRIME
ROS XML config:  /home/jkang/third_party/PRIME_ros/src/prime_contact_id_ros/config/Go2_sim_kunzhao_long_mhe_ros.xml
ROS results:     /home/jkang/third_party/PRIME_ros/results
```

The ROS-local XML is the file to tune for online runs. It still points to the
PRIME experiment descriptions/data by default, but writes outputs in the ROS
workspace.

## Build

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --cmake-args \
  -DPRIME_SOURCE_DIR=/home/jkang/third_party/PRIME \
  -DPRIME_ROS_ENABLE_MULTITHREADS=ON \
  -DPRIME_ROS_NTHREADS=8
source install/setup.bash
```

For a colleague using a different PRIME checkout, replace only
`PRIME_SOURCE_DIR`:

```bash
colcon build --symlink-install \
  --cmake-args \
  -DPRIME_SOURCE_DIR=/path/to/their/PRIME \
  -DPRIME_ROS_ENABLE_MULTITHREADS=ON \
  -DPRIME_ROS_NTHREADS=8
```

The package builds PRIME as a local CMake subdirectory, so PRIME does not need
to be installed globally.

## Run Online Node

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run prime_contact_id_ros prime_moving_window_node \
  --ros-args \
  -p config_path:=/home/jkang/third_party/PRIME_ros/src/prime_contact_id_ros/config/Go2_sim_kunzhao_long_mhe_ros.xml \
  -p solver_verbose:=true
```

The node subscribes to:

```text
/odom
/joint_states
```

It publishes:

```text
/prime_moving_window_node/theta_log_cholesky
/prime_moving_window_node/inertia_dynamic_params
/prime_moving_window_node/solver_stats
```

Use `solver_verbose:=true` to print the FDDP iteration table. Use
`solver_verbose:=false` for quieter online runs.

## Timing And Window Size

The XML uses raw sensor samples for `horizon` and identification knots for
`stride_knots`.

```text
raw_sample_dt       = interval
identification_dt   = down_sample * interval
window_duration     = horizon * interval
shooting_knots      = horizon / down_sample
stride_raw_samples  = stride_knots * down_sample
stride_duration     = stride_knots * down_sample * interval
```

Example with `interval=0.005`, `down_sample=2`, and `stride_knots=50`:

```text
identification_dt  = 0.010 s
stride_raw_samples = 100
stride_duration    = 0.500 s
```

The ROS timer period is `stride_duration`. If a solve is still running when the
next timer fires, the node skips that tick and reports it in `solver_stats`.

## CSV Replay Test

Terminal 1, start the online solver:

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run prime_contact_id_ros prime_moving_window_node \
  --ros-args \
  -p config_path:=/home/jkang/third_party/PRIME_ros/src/prime_contact_id_ros/config/Go2_sim_kunzhao_long_mhe_ros.xml \
  -p solver_verbose:=true
```

Terminal 2, replay the offline CSV as ROS messages:

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run prime_contact_id_ros prime_csv_replay_node \
  --ros-args \
  -p config_path:=/home/jkang/third_party/PRIME_ros/src/prime_contact_id_ros/config/Go2_sim_kunzhao_long_mhe_ros.xml \
  -p start_idx:=10000 \
  -p rate_hz:=200.0
```

Watch outputs:

```bash
ros2 topic echo /prime_moving_window_node/theta_log_cholesky
ros2 topic echo /prime_moving_window_node/inertia_dynamic_params
ros2 topic echo /prime_moving_window_node/solver_stats
```

## Useful Parameters

```text
config_path             XML config used by PRIME
base_odom_topic          default /odom
joint_state_topic        default /joint_states
torque_topic             optional Float64MultiArray effort fallback
online_start_idx         default 0 for live runs
solver_verbose           print FDDP callback table
save_raw_logs            write ros_q_log.csv, ros_v_log.csv, ros_tau_log.csv
clear_results_on_start   clear XML output directory when node starts
```

## Notes For Updating PRIME

If the PRIME checkout changes, rebuild this ROS workspace. If CMake still uses
an old cached path, force a clean configure:

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-clean-cache \
  --cmake-args \
  -DPRIME_SOURCE_DIR=/path/to/updated/PRIME \
  -DPRIME_ROS_ENABLE_MULTITHREADS=ON \
  -DPRIME_ROS_NTHREADS=8
```

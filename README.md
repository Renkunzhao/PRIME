# PRIME ROS 2

PRIME manages the complete Go2 estimation pipeline: IEKF, the input adapter,
the online inertia solver, and the output adapter for Beam GT policies.
`legged_rl_deploy` only consumes the resulting `/prime/inertial_delta`.

## Build

Install `vcstool` once, then import the pinned external dependencies into this
workspace:

```bash
sudo apt install python3-vcstool
cd /home/jkang/third_party/PRIME_ros
vcs import . < dependencies.repos
```

This ROS wrapper (`dev_ros`) uses the core source from `origin/dev` in a
separate checkout. Create it once, then exclude it from colcon discovery:

```bash
git worktree add --detach core origin/dev
touch core/COLCON_IGNORE
```

Build the Unitree layer, Go2 estimator, then PRIME. Replace `humble` with the
installed ROS 2 distribution when needed:

```bash
source /opt/ros/humble/setup.bash

colcon build --packages-up-to unitree_lowlevel \
  --symlink-install --parallel-workers 2 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.bash
colcon build --packages-select unitree_sdk2 \
  --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
colcon build --packages-up-to iekf_go1 \
  --symlink-install --executor sequential --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.bash
colcon build \
  --packages-select prime_contact_id_ros --symlink-install \
  --parallel-workers 1 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

The complete `prime_gt.launch.py` pipeline uses `iekf_go1` from
`src/go2_estimator`.

Confirm that the estimator is available after building:

```bash
ros2 pkg prefix iekf_go1
ros2 pkg executables iekf_go1
```

## Start the estimation pipeline

```bash
WORKSPACE=/home/rkz/code/unitree_ws ros2 launch prime_contact_id_ros prime_gt.launch.py
```

IEKF standard output is written to `simulation_sub-*-stdout.log` in the
launch log directory (normally `~/.ros/log/latest/`). IEKF errors and the
other nodes' terminal output remain visible.

The unified launch uses `/lowstate` and `/iekf/odom`, resamples input at 200 Hz,
and publishes `/prime/odom`, `/prime/joint_states`, and `/prime/inertial_delta`.
The solver also publishes its absolute inertia, loaded-model nominal inertia,
Log-Cholesky parameters, and solver statistics under `/prime_moving_window_node`.

## Tune IEKF from a recorded bag

Run only the recorded `/lowstate` stream and IEKF:

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch prime_contact_id_ros iekf_bag_replay.launch.py
```

The default bag is `data_bag/PRIME-default-4kg-oneside`. Tune the source YAML
at `src/go2_estimator/src/iekf_go1/config/parameters_simulation.yaml`; with the
symlink build, changes are used on the next launch. Useful launch overrides are:

```bash
ros2 launch prime_contact_id_ros iekf_bag_replay.launch.py \
  playback_rate:=0.5 start_offset:=20.0
```

The launch prints IEKF output, publishes the estimate on `/iekf/odom`, and writes
`p_sense.csv`, `v_sense.csv`, and `tau_sense.csv` under
`results/iekf_bag_replay`. It stops when playback finishes. Select another output
directory or preserve existing files with:

```bash
ros2 launch prime_contact_id_ros iekf_bag_replay.launch.py \
  log_directory:=/tmp/iekf_run truncate_logs:=false
```

The CSV layout matches the offline PRIME data. Joint values are in Unitree order
`FR, FL, RR, RL`, so use the existing XML `joint_order` permutation. The IEKF
publishes world-frame linear velocity and body-frame angular velocity. Before
logging, the adapter rotates linear velocity into the body frame expected by
Pinocchio; angular velocity is already in the expected body frame.

The bag has no `/sportmodestate`, so estimator ground-truth comparison fields are
unavailable, but IEKF operation is unaffected.

Visualize the logged motion in MeshCat:

```bash
ros2 run prime_contact_id_ros meshcat_go2_data.py
```

By default, the viewer plays every logged row and then loops the complete motion.

For a shorter or down-sampled preview:

```bash
ros2 run prime_contact_id_ros meshcat_go2_data.py \
  --start-idx 100 --stride 2 --max-frames 500 --no-loop
```

The script can validate paths, dimensions, and joint mapping without opening a
viewer by passing `--check-only`.

## Standalone solver and CSV replay

For already prepared `/odom` and `/joint_states` topics:

```bash
ros2 launch prime_contact_id_ros prime_moving_window.launch.py
```

To replay the CSV specified in the installed XML in another terminal:

```bash
ros2 launch prime_contact_id_ros csv_replay.launch.py
```

Both launch files accept `config_path:=/absolute/path/to/config.xml`.

## Window timing

`interval` is the raw sample period. Identification knots are separated by
`down_sample * interval`; window duration is `horizon * interval` and stride
duration is `stride_knots * down_sample * interval`.
The default configuration uses a 1.25 s window and 0.5 s stride.
It limits the session to 100 windows; adjust `max_windows` for longer sessions.
The solver processes windows in order, so monitor lag in `solver_stats`.
Fresh reception of an estimate does not guarantee a recent measurement window.

## Full recorded-data pipeline

Run bag replay, IEKF, the ROS state adapter, online PRIME, and the inertia output
adapter together with the tuned real-data settings:

```bash
cd /home/jkang/third_party/PRIME_ros
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch prime_contact_id_ros prime_bag_replay.launch.py
```

The launch begins at 22 seconds into `PRIME-default-4kg-oneside`. Override this
with `start_offset:=<seconds>` or select another bag with
`bag_path:=/absolute/path/to/bag`. PRIME receives all q/v/torque samples through
ROS topics and its in-memory buffer; CSV logging is disabled and is not part of
the pipeline. Set `log_csv:=true` only to save an optional diagnostic copy.

The tuned solver uses a 200-sample (1.0 s) raw window, downsamples by two, and
advances 50 identification knots. Its timer therefore starts a new eligible
window every `50 * 2 * 0.005 = 0.5` seconds. FDDP callbacks and CSV logging are
disabled by default. Use `solver_verbose:=true` or `log_csv:=true` only when
diagnosing a run.

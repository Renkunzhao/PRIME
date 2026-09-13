# PRIME ROS 2

PRIME manages the complete Go2 estimation pipeline: IEKF, the input adapter,
the online inertia solver, and the output adapter for Beam GT policies.
`legged_rl_deploy` only consumes the resulting `/prime/inertial_delta`.

## Build

This ROS wrapper (`dev_ros`) uses the core source from `origin/dev` in a
separate checkout. Create it once, then exclude it from colcon discovery:

```bash
git worktree add --detach core origin/dev
touch core/COLCON_IGNORE
```

Build against ROS Jazzy's system Pinocchio and Python:

```bash
source src/unitree_lowlevel/scripts/setup.sh lo jazzy

colcon build \
  --packages-select iekf_go1 prime_contact_id_ros legged_rl_deploy \
  --symlink-install --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
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

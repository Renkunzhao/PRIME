# PRIME → Beam GT policy

The GT-trained ONNX actor receives the **raw 10D inertial delta**, not PRIME's
absolute theta, a mass/inertia tensor, or already normalized observations.
ONNX applies the learned observation normalization internally.

## Data path

```text
/lowstate → IEKF → /iekf/odom
/iekf/odom + /lowstate
  → unitree_prime_state_node.py (200 Hz)
  → /prime/odom + /prime/joint_states
  → prime_moving_window_node
  → /prime_moving_window_node/inertia_dynamic_params (Float64MultiArray)
  + /prime_moving_window_node/nominal_inertia_dynamic_params (Float64MultiArray)
  → prime_inertial_input_node.py
  → /prime/inertial_delta (Float32MultiArray, 10 values)
  → config_gt.yaml external.inertial → ONNX inertial [1,10]
```

`/unitree_go2/inertial_delta` remains simulator GT for comparison; it does not
feed the PRIME configuration. The adapter publishes only upon a valid estimate,
never a startup zero or a heartbeat. The deployment subscriber holds the last
estimate for at most 3 seconds **since reception**. No estimate or a stale
estimate prevents policy inference through the existing external-input checks.
This timeout does not measure the age of PRIME's measurement window.

## Why nominal correction happens before Log-Cholesky

Pinocchio uses additive dynamic parameters in the base frame:

```text
d = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
```

Here `I` is about the **base origin**. PRIME's floating-base inertia includes
fixed child links, while training reads only MuJoCo `base_link`. In the
`Go2_sim_kunzhao_long_mhe` model on PRIME's `origin/dev`, four fixed hip rotors
contribute 4 × 0.089 = 0.356 kg: root nominal 7.277 kg versus training 6.921 kg.
Other PRIME URDFs differ, so the solver publishes its actual loaded-model
nominal once, with transient-local durability for late subscribers. This is
not the first estimated sample and does not depend on the payload present
when PRIME starts.

The adapter uses:

```text
d_policy = d_policy_nominal + (d_prime_estimate - d_prime_nominal)
p = theta(d_policy) - theta(d_policy_nominal)
```

This removes the known, fixed model difference in quantities that add linearly.
For an exact estimate with an added payload, `d_prime_estimate =
d_prime_nominal + d_payload`, hence `d_policy = d_policy_nominal + d_payload`.
The payload remains present, including its first moment and inertia.
**Subtracting two absolute Log-Cholesky vectors across different models does
not implement this correction**, because the transform is nonlinear.

After correcting dynamic parameters, recover COM and COM inertia using the
parallel-axis theorem, then form `S = (0.5*tr(I_com)*identity - I_com)/m`.
Factor `S = U @ U.T` with upper triangular U. The training representation is:

```text
theta = [0.5*log(m), log(U11), log(U22), log(U33), U12, U23, U13, cx, cy, cz]
```

The policy nominal in `prime_inertial_input.yaml` is computed from the compiled
training MJCF, including its inertial quaternion. Recompute it if the training
robot's nominal inertia changes. Do not replace it with an estimated baseline.

This mapping requires PRIME to identify **only the floating-base joint (1)**,
with the same axes and origin as training `base_link`, and to model the same
unloaded robot. It removes known model differences; it does not correct PRIME
estimation bias, a wrong frame, missing moving-link dynamics, or arbitrary
URDF errors. Nonfinite, incorrectly sized, or nonphysical estimates are rejected;
no clipping or additional normalization is applied to p.

## Start all four nodes with one launch

`prime_gt.launch.py` starts the existing IEKF simulation launch, the 200 Hz
state adapter, the PRIME online solver, and the inertial-output adapter.
Ctrl+C stops the group; if any launched process exits, the launch shuts down
the remaining group too. Robot LowState publication and the policy controller
are started using the existing robot setup.

Build instructions are in [the PRIME README](../README.md). Once built:

```bash
source /opt/ros/jazzy/setup.bash
source /home/rkz/code/unitree_ws/install/setup.bash
WORKSPACE=/home/rkz/code/unitree_ws ros2 launch prime_contact_id_ros prime_gt.launch.py
```

PRIME owns the launch, both adapters, their tests, and the policy nominal YAML.
The deployment package only subscribes to `/prime/inertial_delta`.
`WORKSPACE` is required by the IEKF robot model loader.
The unified launch uses no privileged priority or CPU affinity by default;
add `iekf_prefix:="taskset -c 1"` if desired. The standalone IEKF launch keeps
its previous prefix default.

If IEKF is already running, add `start_iekf:=false`.
`prime_config:=/absolute/path/to/config.xml` overrides the installed XML.
CMake generates its local paths from the package and `PRIME_SOURCE_DIR`;
edit `src/prime_contact_id_ros/config/Go2_sim_kunzhao_long_mhe_ros.xml.in`
and rebuild to change defaults.
`inertial_config:=/absolute/path/to/prime_inertial_input.yaml` overrides the
training-model nominal. Its only source copy is in PRIME's package config.

The state adapter subscribes to `/iekf/odom` and `/lowstate`. It uses the IEKF
position, orientation, and body angular velocity, and rotates IEKF world linear
velocity to the body frame with **IEKF's own quaternion**. IEKF aligns initial
yaw, so using the raw LowState IMU quaternion for this conversion would mix world
frames. LowState supplies named joint position, velocity, and measured torque
`tau_est`, in hardware order FR, FL, RR, RL. SportModeState is not needed.

For the current flat-ground experiment, IEKF's corrected z is preserved. It is
not a continuous world-height measurement across changes of support surface.
The 200 Hz adapter matches PRIME XML `interval=0.005` and uses latest-value
resampling without interpolation or strict timestamp synchronization. IEKF now
stamps odometry at publication. The adapter measures freshness from reception
and stamps each output pair with its ROS collection time. If either
input stops updating for over 0.1 s, it stops publishing. This does not reconstruct
sensor acquisition timestamps or guarantee IEKF's internal state is current.

Wait for `/prime/inertial_delta` before activating the GT policy. Keep the
existing depth preprocessing and use `config_gt.yaml` for deployment.

```bash
ros2 topic echo /prime/inertial_delta
ros2 topic hz /prime/inertial_delta
```

The existing solver XML limits the run to 100 windows; adjust `max_windows` for
the session. It processes windows in order, so slow solving can accumulate lag.
Check `solver_stats` and window timing; a freshly received estimate need not be
based on recent measurements. After sensor/time resets or long interruptions,
restart the pipeline to discard the old history.
